#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// actions
#define ACTION_NONE 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_UP 3
#define ACTION_DOWN 4
#define ACTION_JUMP 5

// states
#define state_idle 0 
#define state_walking 1
#define state_sidestepping 2
#define state_crouching 3
#define state_jumping 4
#define state_attacking 5
#define state_blocking 6
#define state_hitstun 7
#define state_knockdown 8
#define state_recovering 9

// collision shapes
#define SHAPE_NONE 0
#define SHAPE_CAPSULE_JOINT 1
#define SHAPE_CAPSULE_OFFSET 2
#define SHAPE_SPHERE_JOINT 3
typedef struct Client Client;

// joints
#define NUM_JOINTS 20
#define J_PELVIS 0 
#define J_LOWER_SPINE 1
#define J_MID_SPINE 2
#define J_UPPER_SPINE 3
#define J_NECK 4
#define J_HEAD 5
#define J_L_CLAV 6
#define J_L_SHOULDER 7
#define J_L_ELBOW 8
#define J_L_WRIST 9
#define J_R_CLAV 10
#define J_R_SHOULDER 11
#define J_R_ELBOW 12
#define J_R_WRIST 13
#define J_L_HIP 14
#define J_L_KNEE 15
#define J_L_ANKLE 16
#define J_R_HIP 17
#define J_R_KNEE 18
#define J_R_ANKLE 19


// only use floats!
typedef struct {
    float score;
    float n; // required as the last field 
} Log;

typedef struct { 
    float x, y, z;
} Vec3;

typedef struct { 
    float w, x, y, z;
} Quat;

typedef struct {
    float m[16];  // Column-major 4x4 matrix
} Mat4;


typedef struct {
    int type;      // Enum: e.g., 0=SHAPE_CAPSULE_JOINT, 1=CAPSULE_OFFSET, 2=SPHERE
    float radius;  // Shared for all types

    union {
        // Capsule between two joints (for limbs; inherits bone orientation from FK)
        struct {
            int joint_a;     // Start joint index
            int joint_b;     // End joint index
            float pad_a;     // Padding offset along bone from A (e.g., 0.1 to shorten)
            float pad_b;     // Padding from B
        } capsule_jnt;

        // Capsule with local offset and orientation (for torso/head; relative to parent joint)
        struct {
            int parent_joint;  // Anchor joint index
            Vec3 local_offset; // Offset from parent (replaces off_x/y/z for Vec3 consistency)
            Quat local_rot;    // Local rotation for the capsule (e.g., to tilt torso)
            float height;      // Capsule length (along its local Y, post-rotation)
        } capsule_off;

        // Sphere at a joint (simple; uses world_pos directly)
        struct {
            int joint;         // Center joint index
            Vec3 local_offset; // Optional local offset from joint (default {0,0,0})
        } sphere;
    };
} CollisionShape;

typedef struct {
    int parent;
    Vec3 local_pos;
    Quat local_rot;
    Vec3 world_pos;
    Quat world_rot;
} Joint;

typedef struct { 
    // ---- root pose ----
    float pos_x, pos_y, pos_z;   // translation
    float facing;                // yaw (whole-body facing)

    // ---- joints ----
    int   num_joints;
    Joint* joints;

    // ---- shapes ----
    int num_shapes;
    CollisionShape *shapes;

    // ---- gameplay ----
    float health;
    int state;
} Character;

typedef struct {
    Log log;                     // required field
    float* observations; // required field. ensure type matches in .py and .c
    int* actions;                // required field. ensure type matches in .py and .c
    float* rewards;              // required field
    unsigned char* terminals;    // required field
    int size;
    int x;
    int goal;
    Character* characters;
    int num_characters;
    Client* client;
} Fighter;

// Vector operations
Vec3 vec3_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 vec3_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 vec3_scale(Vec3 v, float s) { return (Vec3){v.x * s, v.y * s, v.z * s}; }
float vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float vec3_length(Vec3 v) { return sqrtf(vec3_dot(v, v)); }
Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    return len > EPSILON ? vec3_scale(v, 1.0f / len) : v;
}

// Quaternion operations (w is real part, x,y,z imaginary)
Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float ha = angle * 0.5f;
    float s = sinf(ha);
    float c = cosf(ha);
    axis = vec3_normalize(axis);
    return (Quat){c, axis.x * s, axis.y * s, axis.z * s};
}
Quat quat_mul(Quat a, Quat b) {
    return (Quat){
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x
    };
}
Quat quat_inverse(Quat q) {
    float len_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (len_sq > EPSILON) {
        float inv_len_sq = 1.0f / len_sq;
        return (Quat){q.w * inv_len_sq, -q.x * inv_len_sq, -q.y * inv_len_sq, -q.z * inv_len_sq};
    }
    return (Quat){1.0f, 0.0f, 0.0f, 0.0f};
}
Vec3 quat_rotate_vec(Quat q, Vec3 v) {
    Quat vq = {0.0f, v.x, v.y, v.z};
    Quat res = quat_mul(quat_mul(q, vq), quat_inverse(q));
    return (Vec3){res.x, res.y, res.z};
}

// Matrix operations (identity, multiply, from quat+pos)
void mat4_identity(Mat4 *m) {
    for (int i = 0; i < 16; i++) m->m[i] = 0.0f;
    m->m[0] = m->m[5] = m->m[10] = m->m[15] = 1.0f;
}
void mat4_mul(Mat4 a, Mat4 b, Mat4 *res) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            res->m[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                res->m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
        }
    }
}
void mat4_from_quat_pos(Mat4 *m, Quat q, Vec3 p) {
    float xx = q.x * q.x, xy = q.x * q.y, xz = q.x * q.z, xw = q.x * q.w;
    float yy = q.y * q.y, yz = q.y * q.z, yw = q.y * q.w;
    float zz = q.z * q.z, zw = q.z * q.w;

    m->m[0] = 1 - 2 * (yy + zz); m->m[1] = 2 * (xy - zw); m->m[2] = 2 * (xz + yw); m->m[3] = 0;
    m->m[4] = 2 * (xy + zw); m->m[5] = 1 - 2 * (xx + zz); m->m[6] = 2 * (yz - xw); m->m[7] = 0;
    m->m[8] = 2 * (xz - yw); m->m[9] = 2 * (yz + xw); m->m[10] = 1 - 2 * (xx + yy); m->m[11] = 0;
    m->m[12] = p.x; m->m[13] = p.y; m->m[14] = p.z; m->m[15] = 1;
}


void init_skeleton(Character* c){
    
    for (int i = 0; i < c->num_joints; i++) {
        c->joints[i].local_rot = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        c->joints[i].world_rot = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        c->joints[i].world_pos = (Vec3){0.0f, 0.0f, 0.0f};
    }

    c->joints[J_PELVIS].parent = -1;
    c->joints[J_PELVIS].local_pos = (Vec3){0.0f, 0.9f, 0.0f};

    c->joints[J_LOWER_SPINE].parent = J_PELVIS;
    c->joints[J_LOWER_SPINE].local_pos = (Vec3){0.0f, 0.2f, 0.0f};

    c->joints[J_MID_SPINE].parent = J_LOWER_SPINE;
    c->joints[J_MID_SPINE].local_pos = (Vec3){0.0f, 0.2f, 0.0f};

    c->joints[J_UPPER_SPINE].parent = J_MID_SPINE;
    c->joints[J_UPPER_SPINE].local_pos = (Vec3){0.0f, 0.2f, 0.0f};

    c->joints[J_NECK].parent = J_UPPER_SPINE;
    c->joints[J_NECK].local_pos = (Vec3){0.0f, 0.15f, 0.0f};

    c->joints[J_HEAD].parent = J_NECK;
    c->joints[J_HEAD].local_pos = (Vec3){0.0f, 0.15f, 0.0f};

    c->joints[J_L_CLAV].parent = J_UPPER_SPINE;
    c->joints[J_L_CLAV].local_pos = (Vec3){-0.1f, 0.05f, 0.0f};

    c->joints[J_L_SHOULDER].parent = J_L_CLAV;
    c->joints[J_L_SHOULDER].local_pos = (Vec3){-0.15f, 0.0f, 0.0f};

    c->joints[J_L_ELBOW].parent = J_L_SHOULDER;
    c->joints[J_L_ELBOW].local_pos = (Vec3){-0.25f, 0.0f, 0.0f};

    c->joints[J_L_WRIST].parent = J_L_ELBOW;
    c->joints[J_L_WRIST].local_pos = (Vec3){-0.25f, 0.0f, 0.0f};

    c->joints[J_R_CLAV].parent = J_UPPER_SPINE;
    c->joints[J_R_CLAV].local_pos = (Vec3){0.1f, 0.05f, 0.0f};

    c->joints[J_R_SHOULDER].parent = J_R_CLAV;
    c->joints[J_R_SHOULDER].local_pos = (Vec3){0.15f, 0.0f, 0.0f};

    c->joints[J_R_ELBOW].parent = J_R_SHOULDER;
    c->joints[J_R_ELBOW].local_pos = (Vec3){0.25f, 0.0f, 0.0f};

    c->joints[J_R_WRIST].parent = J_R_ELBOW;
    c->joints[J_R_WRIST].local_pos = (Vec3){0.25f, 0.0f, 0.0f};

    c->joints[J_L_HIP].parent = J_PELVIS;
    c->joints[J_L_HIP].local_pos = (Vec3){-0.1f, -0.05f, 0.0f};

    c->joints[J_L_KNEE].parent = J_L_HIP;
    c->joints[J_L_KNEE].local_pos = (Vec3){0.0f, -0.45f, 0.0f};

    c->joints[J_L_ANKLE].parent = J_L_KNEE;
    c->joints[J_L_ANKLE].local_pos = (Vec3){0.0f, -0.45f, 0.0f};

    c->joints[J_R_HIP].parent = J_PELVIS;
    c->joints[J_R_HIP].local_pos = (Vec3){0.1f, -0.05f, 0.0f};

    c->joints[J_R_KNEE].parent = J_R_HIP;
    c->joints[J_R_KNEE].local_pos = (Vec3){0.0f, -0.45f, 0.0f};

    c->joints[J_R_ANKLE].parent = J_R_KNEE;
    c->joints[J_R_ANKLE].local_pos = (Vec3){0.0f, -0.45f, 0.0f};
}

void init_shapes(Character *c) {
    float bone_radius = 0.05f;
    int shape_idx = 0;
    // Spine chain
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;  // CAPSULE_JOINT
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_PELVIS;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_LOWER_SPINE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_LOWER_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_MID_SPINE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_MID_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_UPPER_SPINE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_UPPER_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_NECK;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_NECK;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_HEAD;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    // Left arm
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_UPPER_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_CLAV;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_L_CLAV;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_SHOULDER;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_L_SHOULDER;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_ELBOW;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_L_ELBOW;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_WRIST;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    // Right arm
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_UPPER_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_CLAV;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_R_CLAV;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_SHOULDER;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_R_SHOULDER;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_ELBOW;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_R_ELBOW;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_WRIST;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    // Left leg
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_PELVIS;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_HIP;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_L_HIP;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_KNEE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_L_KNEE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_L_ANKLE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    // Right leg
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_PELVIS;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_HIP;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_R_HIP;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_KNEE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_R_KNEE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_R_ANKLE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;
}

void compute_fk(Joint *joints, int joint_idx, Quat parent_rot, Vec3 parent_pos) {
    Joint *j = &joints[joint_idx];
    j->world_rot = quat_mul(parent_rot, j->local_rot);
    Vec3 local_offset = quat_rotate_vec(parent_rot, j->local_pos);
    j->world_pos = vec3_add(parent_pos, local_offset);

    // Recurse to children (hardcoded for your 20-joint hierarchy)
    if (joint_idx == J_PELVIS) {  // Root: Pelvis
        compute_fk(joints, J_LOWER_SPINE, j->world_rot, j->world_pos);
        compute_fk(joints, J_L_HIP, j->world_rot, j->world_pos);
        compute_fk(joints, J_R_HIP, j->world_rot, j->world_pos);
    } else if (joint_idx == J_LOWER_SPINE) {
        compute_fk(joints, J_MID_SPINE, j->world_rot, j->world_pos);
    } else if (joint_idx == J_MID_SPINE) {
        compute_fk(joints, J_UPPER_SPINE, j->world_rot, j->world_pos);
    } else if (joint_idx == J_UPPER_SPINE) {
        compute_fk(joints, J_NECK, j->world_rot, j->world_pos);
        compute_fk(joints, J_L_CLAV, j->world_rot, j->world_pos);
        compute_fk(joints, J_R_CLAV, j->world_rot, j->world_pos);
    } else if (joint_idx == J_NECK) {
        compute_fk(joints, J_HEAD, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_CLAV) {
        compute_fk(joints, J_L_SHOULDER, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_SHOULDER) {
        compute_fk(joints, J_L_ELBOW, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_ELBOW) {
        compute_fk(joints, J_L_WRIST, j->world_rot, j->world_pos);
    } else if (joint_idx == J_R_CLAV) {
        compute_fk(joints, J_R_SHOULDER, j->world_rot, j->world_pos);
    } else if (joint_idx == J_R_SHOULDER) {
        compute_fk(joints, J_R_ELBOW, j->world_rot, j->world_pos);
    } else if (joint_idx == J_R_ELBOW) {
        compute_fk(joints, J_R_WRIST, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_HIP) {
        compute_fk(joints, J_L_KNEE, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_KNEE) {
        compute_fk(joints, J_L_ANKLE, j->world_rot, j->world_pos);
    } else if (joint_idx == J_R_HIP) {
        compute_fk(joints, J_R_KNEE, j->world_rot, j->world_pos);
    } else if (joint_idx == J_R_KNEE) {
        compute_fk(joints, J_R_ANKLE, j->world_rot, j->world_pos);
    }
    // No recursion for leaf nodes like head, wrists, ankles
}

// Usage: After updating local_rots or init, call compute_fk(c->joints, J_PELVIS, (Quat){1.0f, 0.0f, 0.0f, 0.0f}, (Vec3){c->pos_x, c->pos_y, c->pos_z});

void init(Fighter* env) {
    env->num_characters = 2;
    env->characters = (Character*)calloc(env->num_characters, sizeof(Character));
    for (int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        c->health = 100;
        c->pos_x = (i == 0) ? -5.0f : 5.0f; // spawn left/right
        c->pos_y = 1.0f;
        c->pos_z = 0.0f;
        c->facing = (i == 0) ? -PI/2.0f : PI/2.0f;
        c->state  = 0;
        c->num_shapes = 19;  
        c->shapes = calloc(c->num_shapes, sizeof(CollisionShape));
        c->num_joints = NUM_JOINTS;
        c->joints  = calloc(NUM_JOINTS, sizeof(Joint));;
        init_skeleton(c); // joints + hierarchy
        init_shapes(c);   // collision capsules/spheres
        compute_fk(c->joints, J_PELVIS, (Quat){1.0f, 0.0f, 0.0f, 0.0f}, (Vec3){c->pos_x, c->pos_y, c->pos_z});
    }
    printf("init\n");
}

void c_reset(Fighter* env) {
    for (int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        c->health = 100;
        c->pos_x = (i == 0) ? -5.0f : 5.0f; // spawn left/right
        c->pos_y = 0.0f;
        c->pos_z = 0.0f;
        c->facing = (i==0) ? -PI/2.0f : PI/2.0f;
        c->state  = 0;

        init_skeleton(c); // joints + hierarchy
        init_shapes(c);   // collision capsules/spheres
        compute_fk(c->joints, J_PELVIS, (Quat){1.0f, 0.0f, 0.0f, 0.0f}, (Vec3){c->pos_x, c->pos_y, c->pos_z});
    }

    printf("reset\n");
}

void sidestep(Fighter* env, Character* character, float direction, int target_index) {
    float sidestep_radius = 4.0f;
    float sidestep_speed = 0.05f;
    float cos_facing = cos(character->facing);
    float sin_facing = sin(character->facing);
    // calculate current position relative to circle center
    float center_x = character->pos_x + sidestep_radius * cos_facing;  // circle center x
    float center_z = character->pos_z + sidestep_radius * sin_facing;  // circle center z
    
    // update facing for circular motion
    character->facing += sidestep_speed * direction;
    float new_cos_facing = cos(character->facing);
    float new_sin_facing = sin(character->facing);
    // calculate new position on circle circumference
    character->pos_x = (center_x) - (sidestep_radius * new_cos_facing);
    character->pos_z = (center_z) - (sidestep_radius * new_sin_facing);
    // new facing toward target
    character->facing = atan2(env->characters[target_index].pos_z - character->pos_z, env->characters[target_index].pos_x - character->pos_x);
}

/*void move_character(Fighter* env, int character_index, int target_index, int action) {
    Character* character = &env->characters[character_index];
    if (action == action_left) {
        character->pos_x -= 0.15 * cos(character->facing);
        character->pos_z -= 0.15 * sin(character->facing);
    } else if (action == action_right) {
        character->pos_x += 0.15 * cos(character->facing);
        character->pos_z += 0.15 * sin(character->facing);
    } else if (action == action_up){
        sidestep(env, character, 1, target_index);
    } else if (action == action_down){
        sidestep(env, character, -1, target_index);
    } 
}


*/
void c_step(Fighter* env) {
    for(int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;
        Quat yaw_rot = quat_from_axis_angle((Vec3){0.0f, 1.0f, 0.0f}, c->facing);
        compute_fk(c->joints, J_PELVIS, yaw_rot, (Vec3){c->pos_x, c->pos_y, c->pos_z});
    }
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client* make_client(Fighter* env){    
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "pufferlib fighter");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/puffers_128.png");
    
    client->default_camera_position = (Vector3){ 
        0,           // same x as target
        10.0f,   // 20 units above target
        20.0f    // 20 units behind target
    };
    client->default_camera_target = (Vector3){0, 0, 0};
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };  // y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    //DisableCursor(); 
    return client;
}

/*void UpdateCameraFighter(Fighter* env, Client* client){
    Character* f1 = &env->characters[0];
    Character* f2 = &env->characters[1];
    Vector3 pos1 = (Vector3){f1->pos_x, f1->pos_y, f1->pos_z};
    Vector3 pos2 = (Vector3){f2->pos_x, f2->pos_y, f2->pos_z};
   
    // midpoint
    Vector3 midpoint = Vector3Scale(Vector3Add(pos1,pos2), 0.5f);
    // dir
    Vector3 dir = Vector3Normalize(Vector3Subtract(pos2,pos1));
    // perpendicular
    Vector3 up = (Vector3){0,-1,0};
    Vector3 side = Vector3Normalize(Vector3CrossProduct(up,dir));
    // dynamic zoom
    float distance = Vector3Distance(pos1,pos2);
    float zoom = Lerp(4.0f, 15.0f, Clamp(distance / 15.0f, 0.0f, 1.0f));
    // camera position
    Vector3 offset = Vector3Scale(side,zoom);
    offset.y = 5.0f;
    Vector3 next_pos = Vector3Add(midpoint, offset);
    // interpolate
    float smooth = 0.1f;
    client->camera.position = Vector3Lerp(client->camera.position, next_pos, smooth);
    client->camera.target = Vector3Lerp(client->camera.target, midpoint, smooth);    
}*/

void c_render(Fighter* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    //UpdateCameraFighter(env, client);
    UpdateCamera(&client->camera, CAMERA_FREE);
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);

    // draw a simple ground plane for reference (xz plane at y=0)
    DrawPlane((Vector3){0.0f, 0.0f, 0.0f}, (Vector2){20.0f, 20.0f}, LIGHTGRAY);

    // draw grid for depth perception
    DrawGrid(20, 1.0f);
    // render each character's hurt_shapes as capsules (visualize the fighters)
    //
    for (int ci = 0; ci < env->num_characters; ci++) {
        Character *chara = &env->characters[ci];
        Color fighter_color = (ci == 0) ? RED : BLUE;

        for (int i = 0; i < chara->num_shapes; i++) {
            CollisionShape *s = &chara->shapes[i];
            if(i ==2) fighter_color = PINK;
            if(i ==3) fighter_color = GREEN;
            if(i ==4) fighter_color = YELLOW;
            if(i ==5) fighter_color = PURPLE;
            if(i ==6) fighter_color = GRAY;
            if(i ==7) fighter_color = BLACK;
            if(i ==8) fighter_color = ORANGE;
            if(i ==9) fighter_color = BROWN;
            if(i ==10) fighter_color = VIOLET;
            if(i ==11) fighter_color = PUFF_CYAN;
            if(i ==12) fighter_color = LIME;
            if(i ==13) fighter_color = BEIGE;

            if (s->type == SHAPE_CAPSULE_JOINT) {
                int a = s->capsule_jnt.joint_a;
                int b = s->capsule_jnt.joint_b;

                Vec3 A_base = chara->joints[a].world_pos;
                Vec3 B_base = chara->joints[b].world_pos;
                Vector3 dir = Vector3Normalize((Vector3){B_base.x - A_base.x, B_base.y - A_base.y, B_base.z - A_base.z});
                Vector3 dir_inv = Vector3Scale(dir, -1.0f);

                // Apply padding
                Vector3 A = (Vector3){A_base.x + dir.x * s->capsule_jnt.pad_a, A_base.y + dir.y * s->capsule_jnt.pad_a, A_base.z + dir.z * s->capsule_jnt.pad_a};
                Vector3 B = (Vector3){B_base.x + dir_inv.x * s->capsule_jnt.pad_b, B_base.y + dir_inv.y * s->capsule_jnt.pad_b, B_base.z + dir_inv.z * s->capsule_jnt.pad_b};

                DrawCapsule(A, B, s->radius, 16, 8, fighter_color);
                DrawCapsuleWires(A, B, s->radius, 16, 8, BLACK);
            }

            else if (s->type == SHAPE_CAPSULE_OFFSET) {
                int p = s->capsule_off.parent_joint;
                Vec3 P = chara->joints[p].world_pos;

                Quat parent_rot = chara->joints[p].world_rot;
                Quat world_rot = quat_mul(parent_rot, s->capsule_off.local_rot);

                Vec3 rotated_offset = quat_rotate_vec(world_rot, s->capsule_off.local_offset);
                Vec3 base = vec3_add(P, rotated_offset);

                Vec3 height_dir = quat_rotate_vec(world_rot, (Vec3){0.0f, s->capsule_off.height / 2.0f, 0.0f});
                Vector3 A = (Vector3){base.x - height_dir.x, base.y - height_dir.y, base.z - height_dir.z};
                Vector3 B = (Vector3){base.x + height_dir.x, base.y + height_dir.y, base.z + height_dir.z};

                DrawCapsule(A, B, s->radius, 16, 8, fighter_color);
                DrawCapsuleWires(A, B, s->radius, 16, 8, BLACK);
            }

            else if (s->type == SHAPE_SPHERE_JOINT) {
                int j = s->sphere.joint;
                Vec3 J_base = chara->joints[j].world_pos;

                Quat joint_rot = chara->joints[j].world_rot;
                Vec3 rotated_offset = quat_rotate_vec(joint_rot, s->sphere.local_offset);
                Vector3 J = (Vector3){J_base.x + rotated_offset.x, J_base.y + rotated_offset.y, J_base.z + rotated_offset.z};

                DrawSphere(J, s->radius, fighter_color);
            }
        }
    }
    EndMode3D();
    
    // draw health bars on top of screen (2d overlay)
    float health_bar_width = 200.0f;
    float health_bar_height = 20.0f;
    float health_bar_y = 20.0f;
    float health_bar_margin = 20.0f;
    
    // character 1 health bar (top left)
    float char1_x = health_bar_margin;
    float char1_health_ratio = env->characters[0].health / 100.0f;
    Rectangle char1_background = {char1_x, health_bar_y, health_bar_width, health_bar_height};
    Rectangle char1_health = {char1_x, health_bar_y, health_bar_width * char1_health_ratio, health_bar_height};
    DrawRectangleRec(char1_background, DARKGRAY);
    DrawRectangleRec(char1_health, RED);
    DrawRectangleLinesEx(char1_background, 2, WHITE);
    DrawText(TextFormat("player 1: %d", (int)env->characters[0].health), char1_x + 10, health_bar_y + 25, 20, WHITE);
    
    // character 2 health bar (top right)
    float char2_x = client->width - health_bar_width - health_bar_margin;
    float char2_health_ratio = env->characters[1].health / 100.0f;
    Rectangle char2_background = {char2_x, health_bar_y, health_bar_width, health_bar_height};
    Rectangle char2_health = {char2_x, health_bar_y, health_bar_width * char2_health_ratio, health_bar_height};
    DrawRectangleRec(char2_background, DARKGRAY);
    DrawRectangleRec(char2_health, BLUE);
    DrawRectangleLinesEx(char2_background, 2, WHITE);
    DrawText(TextFormat("player 2: %d", (int)env->characters[1].health), char2_x + 10, health_bar_y + 25, 20, WHITE);
    EndDrawing();
}

void c_close(Fighter* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
