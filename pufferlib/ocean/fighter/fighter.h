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
#define ACTION_SIDESTEP_UP 3
#define ACTION_SIDESTEP_DOWN 4
#define ACTION_CROUCH 5
#define ACTION_LIGHT_PUNCH 6
#define ACTION_LOW_KICK 7
#define ACTION_MEDIUM_KICK 8
#define ACTION_MEDIUM_PUNCH 9

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
#define NUM_JOINTS 17
#define J_PELVIS 0 
#define J_R_HIP 1
#define J_R_KNEE 2
#define J_R_ANKLE 3
#define J_L_HIP 4
#define J_L_KNEE 5
#define J_L_ANKLE 6
#define J_SPINE 7
#define J_THORAX 8
#define J_NECK 9
#define J_HEAD 10
#define J_L_SHOULDER 11
#define J_L_ELBOW 12
#define J_L_WRIST 13
#define J_R_SHOULDER 14
#define J_R_ELBOW 15
#define J_R_WRIST 16
// Gizmo types
#define GIZMO_AXIS_X 0
#define GIZMO_AXIS_Y 1
#define GIZMO_AXIS_Z 2
#define GIZMO_AXIS_NONE -1

// only use floats!
typedef struct {
    float score;
    float perf;
    float episode_length;
    float episode_return;
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
    int move_frame_count;
    int move_frame_end;
    int move_frame_start;
    int motion_end_frame;
    float total_root_distance;
    float* joint_move_x;
    float* joint_move_y;
    float* joint_move_z;
    Quat* local_rot;
    float scale;
} Move;

typedef struct { 
    // ---- root pose ----
    float pos_x, pos_y, pos_z;   // translation
    float facing;                // yaw (whole-body facing)

    // ---- joints ----
    int   num_joints;
    Joint* joints;
    float* bone_lengths;

    // ---- shapes ----
    int num_shapes;
    CollisionShape *shapes;
    CollisionShape* push_shapes;
    // animation
    int active_animation_idx;
    int anim_timestep;
    int anim_total_frames;
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
    int tick;
    Character* characters;
    int num_characters;
    int human_agent_idx;
    Move* moveset;
    int num_moves;
    Client* client;
} Fighter;

// Load moveset
void load_moveset(const char* filename, Fighter* env){
    FILE* file = fopen(filename, "rb");
    if(!file) return;
    fread(&env->num_moves, sizeof(int), 1, file);
    env->moveset = (Move*)malloc(env->num_moves * sizeof(Move));
    for(int i = 0; i < env->num_moves; i++){
        fread(&env->moveset[i].move_frame_count, sizeof(int), 1, file);
        int frame_count = env->moveset[i].move_frame_count;
        printf("frame count: %d\n", frame_count);
        env->moveset[i].joint_move_x = (float*)malloc(frame_count*17*sizeof(float));
        env->moveset[i].joint_move_y = (float*)malloc(frame_count*17*sizeof(float));
        env->moveset[i].joint_move_z = (float*)malloc(frame_count*17*sizeof(float));
        for(int f_idx = 0; f_idx < frame_count; f_idx++){
            fread(&env->moveset[i].joint_move_x[f_idx*17], sizeof(float), 17, file);
            fread(&env->moveset[i].joint_move_y[f_idx*17], sizeof(float), 17, file);
            fread(&env->moveset[i].joint_move_z[f_idx*17], sizeof(float), 17, file);
        }
    }
    fclose(file);
}

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

Quat rotation_between_vectors(Vec3 a, Vec3 b) {
    a = vec3_normalize(a);
    b = vec3_normalize(b);
    float dot = vec3_dot(a,b);
    if (dot < -0.9999f) {
        // 180°: pick any axis orthogonal to a
        Vec3 axis = fabsf(a.x) < 0.1f ? vec3_normalize(vec3_cross(a, (Vec3){1,0,0}))
                                      : vec3_normalize(vec3_cross(a, (Vec3){0,1,0}));
        return quat_from_axis_angle(axis, PI);
    }
    Vec3 cross = vec3_cross(a,b);
    Quat q;
    q.w = 1.0f + dot;
    q.x = cross.x;
    q.y = cross.y;
    q.z = cross.z;
    
    float len = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (len > 1e-8f){
        q.w /= len;
        q.x /= len;
        q.y /= len;
        q.z /= len;
    } else {
        q = (Quat){1,0,0,0};
    }
    return q;
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

// Quaternion slerp (for interpolation; helper for splines)
Quat quat_slerp(Quat a, Quat b, float t) {
    float dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (dot < 0) {  // Shortest path
        b = (Quat){-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }
    if (dot > 0.9995f) {  // Linear if nearly same
        return (Quat){
            a.w + t * (b.w - a.w),
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)
        };
    }
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    return (Quat){
        sinf((1 - t) * theta) / sin_theta * a.w + sinf(t * theta) / sin_theta * b.w,
        sinf((1 - t) * theta) / sin_theta * a.x + sinf(t * theta) / sin_theta * b.x,
        sinf((1 - t) * theta) / sin_theta * a.y + sinf(t * theta) / sin_theta * b.y,
        sinf((1 - t) * theta) / sin_theta * a.z + sinf(t * theta) / sin_theta * b.z
    };
}

// Quaternion log and exp for tangent space (for Catmull-Rom)
Vec3 quat_log(Quat q) {
    float len = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
    if (len < EPSILON) return (Vec3){0,0,0};
    float angle = acosf(q.w) * 2.0f;
    return vec3_scale((Vec3){q.x / len, q.y / len, q.z / len}, angle);
}

Quat quat_exp(Vec3 v) {
    float len = vec3_length(v);
    if (len < EPSILON) return (Quat){1, 0, 0, 0};
    float ha = len / 2.0f;
    float s = sinf(ha);
    return (Quat){cosf(ha), v.x * s / len, v.y * s / len, v.z * s / len};
}

// Catmull-Rom spline for quaternions (u in [0,1] between p1 and p2; needs 4 keys)
Quat catmull_rom_quat(Quat p0, Quat p1, Quat p2, Quat p3, float u) {
    // Tangent space relative to p1
    Vec3 t0 = quat_log(quat_mul(quat_inverse(p1), p0));
    Vec3 t1 = (Vec3){0,0,0};
    Vec3 t2 = quat_log(quat_mul(quat_inverse(p1), p2));
    Vec3 t3 = quat_log(quat_mul(quat_inverse(p1), p3));

    // Catmull-Rom hermite in tangent space (tension alpha=0.5)
    float u2 = u * u, u3 = u2 * u;
    Vec3 h1 = vec3_scale(t1, 2*u3 - 3*u2 + 1);
    Vec3 m1 = vec3_scale(vec3_sub(t2, t0), 0.5f * (u3 - 2*u2 + u));
    Vec3 h3 = vec3_scale(t2, -2*u3 + 3*u2);
    Vec3 m2 = vec3_scale(vec3_sub(t3, t1), 0.5f * (u3 - u2));

    Vec3 tangent = vec3_add(vec3_add(h1, m1), vec3_add(h3, m2));
    return quat_mul(p1, quat_exp(tangent));
}

static inline Vec3 to_engine(Vec3 p){
    return (Vec3){p.x, -p.y, -p.z};
}

static inline void build_primary_child_map(int out_child[NUM_JOINTS]) {
    for (int i=0;i<NUM_JOINTS;i++) out_child[i] = -1;
    out_child[J_PELVIS]     = J_SPINE;
    out_child[J_SPINE]      = J_THORAX;
    out_child[J_THORAX]     = J_NECK;
    out_child[J_NECK]       = J_HEAD;

    out_child[J_L_SHOULDER] = J_L_ELBOW;
    out_child[J_L_ELBOW]    = J_L_WRIST;

    out_child[J_R_SHOULDER] = J_R_ELBOW;
    out_child[J_R_ELBOW]    = J_R_WRIST;

    out_child[J_L_HIP]      = J_L_KNEE;
    out_child[J_L_KNEE]     = J_L_ANKLE;

    out_child[J_R_HIP]      = J_R_KNEE;
    out_child[J_R_KNEE]     = J_R_ANKLE;
}

// Parent-first traversal order
static const int ORDER[NUM_JOINTS] = {
    J_PELVIS, J_SPINE, J_THORAX, J_NECK, J_HEAD,
    J_L_HIP, J_L_KNEE, J_L_ANKLE,
    J_R_HIP, J_R_KNEE, J_R_ANKLE,
    J_L_SHOULDER, J_L_ELBOW, J_L_WRIST,
    J_R_SHOULDER, J_R_ELBOW, J_R_WRIST
};

void compute_local_rotations(Move* move, Character* c) {
    move->local_rot = (Quat*)malloc(sizeof(Quat) * move->move_frame_count * NUM_JOINTS);

    int primary_child[NUM_JOINTS];
    build_primary_child_map(primary_child);

    // Precompute each joint’s rest “bone” (in parent local): vector from parent -> primary child
    Vec3 bone_rest_local[NUM_JOINTS];  // in parent space
    for (int p = 0; p < NUM_JOINTS; ++p) {
        int child = primary_child[p];
        if (child < 0) { bone_rest_local[p] = (Vec3){0,0,1}; continue; } 
        bone_rest_local[p] = vec3_normalize(c->joints[child].local_pos);
    }

    // For each frame, solve local rotations top-down
    for (int f = 0; f < move->move_frame_count; f++) {
        // 1) mocap positions in engine coords
        Vec3 mpos[NUM_JOINTS];
        for (int j = 0; j < NUM_JOINTS; j++) {
            mpos[j] = to_engine((Vec3){
                move->joint_move_x[f*NUM_JOINTS + j],
                move->joint_move_y[f*NUM_JOINTS + j],
                move->joint_move_z[f*NUM_JOINTS + j]
            });
        }

        // 2) temp world rotations for this frame (parent-first)
        Quat world_rot[NUM_JOINTS];

        for (int oi = 0; oi < NUM_JOINTS; oi++) {
            int p = ORDER[oi];
            int par = c->joints[p].parent;
            Quat parent_world = (par < 0) ? (Quat){1,0,0,0} : world_rot[par];

            // default: identity (for leaves)
            Quat local = (Quat){1,0,0,0};

            int child = primary_child[p];
            if (child >= 0) {
                // Desired bone direction in WORLD this frame:
                Vec3 bone_curr_world = vec3_normalize(vec3_sub(mpos[child], mpos[p]));
                // Bring that into the PARENT's local space:
                Vec3 target_in_parent =
                    (par < 0) ? bone_curr_world
                              : quat_rotate_vec(quat_inverse(parent_world), bone_curr_world);
                // Rotate parent’s rest bone into this target (both in parent local space):
                local = rotation_between_vectors(bone_rest_local[p], target_in_parent);
            }

            // Store solved local rotation on the PARENT joint (correct place)
            move->local_rot[f*NUM_JOINTS + p] = local;

            // Propagate world rotation for descendants
            world_rot[p] = quat_mul(parent_world, local);
        }
    }
}

void align_skeleton_to_mocap(Character *c, Move *move) {
    Vec3 mocap_rest[NUM_JOINTS], mocap_dir[NUM_JOINTS];
    for (int j = 0; j < NUM_JOINTS; j++) {
        mocap_rest[j] = to_engine((Vec3){
            move->joint_move_x[j],
            move->joint_move_y[j],
            move->joint_move_z[j]
        });
    }
    for (int j = 0; j < NUM_JOINTS; j++) {
        int p = c->joints[j].parent;
        if (p < 0) continue;
        Vec3 dir = vec3_sub(mocap_rest[j], mocap_rest[p]);
        Vec3 unit_dir = vec3_normalize(dir);
        Vec3 target_pos = vec3_scale(unit_dir, c->bone_lengths[j]);
        c->joints[j].local_pos = target_pos;
    } 
}

void init_skeleton(Character* c){
    
    for (int i = 0; i < c->num_joints; i++) {
        c->joints[i].local_rot = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        c->joints[i].world_rot = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        c->joints[i].world_pos = (Vec3){0.0f, 0.0f, 0.0f};
    }

    c->joints[J_PELVIS].parent = -1;
    c->joints[J_PELVIS].local_pos = (Vec3){0.0f, 1.0f, 0.0f};

    c->joints[J_SPINE].parent = J_PELVIS;
    c->joints[J_SPINE].local_pos = (Vec3){0.0f, 0.3f, 0.0f};

    c->joints[J_THORAX].parent = J_SPINE;
    c->joints[J_THORAX].local_pos = (Vec3){0.0f, 0.3f, 0.0f};

    c->joints[J_NECK].parent = J_THORAX;
    c->joints[J_NECK].local_pos = (Vec3){0.0f, 0.15f, 0.0f};

    c->joints[J_HEAD].parent = J_NECK;
    c->joints[J_HEAD].local_pos = (Vec3){0.0f, 0.15f, 0.0f};

    c->joints[J_L_SHOULDER].parent = J_THORAX;
    c->joints[J_L_SHOULDER].local_pos = (Vec3){-0.15f, -0.05f, 0.0f};

    c->joints[J_L_ELBOW].parent = J_L_SHOULDER;
    c->joints[J_L_ELBOW].local_pos = (Vec3){-0.25f, 0.0f, 0.0f};

    c->joints[J_L_WRIST].parent = J_L_ELBOW;
    c->joints[J_L_WRIST].local_pos = (Vec3){-0.25f, 0.0f, 0.0f};

    c->joints[J_R_SHOULDER].parent = J_THORAX;
    c->joints[J_R_SHOULDER].local_pos = (Vec3){0.15f, -0.05f, 0.0f};

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

void store_bone_lengths(Character* c){
    for (int j = 0; j < NUM_JOINTS; j++) {
        int p = c->joints[j].parent;
        if (p < 0) { c->bone_lengths[j] = 0; continue; }
        c->bone_lengths[j] = vec3_length(c->joints[j].local_pos);
    }
}

void init_shapes(Character *c) {
    float bone_radius = 0.05f;
    int shape_idx = 0;
    // Spine chain
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;  // CAPSULE_JOINT
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_PELVIS;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_SPINE;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_SPINE;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_THORAX;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_THORAX;
    c->shapes[shape_idx].capsule_jnt.joint_b = J_NECK;
    c->shapes[shape_idx].capsule_jnt.pad_a = 0.0f;
    c->shapes[shape_idx].capsule_jnt.pad_b = 0.0f;
    shape_idx++;

    c->shapes[shape_idx].type = SHAPE_SPHERE_JOINT;
    c->shapes[shape_idx].radius = 0.1;
    c->shapes[shape_idx].sphere.joint = J_NECK;
    c->shapes[shape_idx].sphere.local_offset = (Vec3){0.0, 0.15, 0.0};
    shape_idx++;

    // Left arm
    c->shapes[shape_idx].type = SHAPE_CAPSULE_JOINT;
    c->shapes[shape_idx].radius = bone_radius;
    c->shapes[shape_idx].capsule_jnt.joint_a = J_THORAX;
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
    c->shapes[shape_idx].capsule_jnt.joint_a = J_THORAX;
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

    int p_idx = 0;
    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.4;
    c->push_shapes[p_idx].sphere.joint = J_SPINE;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.25;
    c->push_shapes[p_idx].sphere.joint = J_THORAX;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

 
    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_L_ELBOW;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_R_ELBOW;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;
    
    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_L_KNEE;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_R_KNEE;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_L_ANKLE;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;

    c->push_shapes[p_idx].type = SHAPE_SPHERE_JOINT;
    c->push_shapes[p_idx].radius = 0.1;
    c->push_shapes[p_idx].sphere.joint = J_R_ANKLE;
    c->push_shapes[p_idx].sphere.local_offset = (Vec3){0.0, 0.0, 0.0};
    p_idx++;
}

void compute_fk(Joint *joints, int joint_idx, Quat parent_rot, Vec3 parent_pos) {
    Joint *j = &joints[joint_idx];
    j->world_rot = quat_mul(parent_rot, j->local_rot);
    Vec3 local_offset = quat_rotate_vec(parent_rot, j->local_pos);
    j->world_pos = vec3_add(parent_pos, local_offset);

    // Recurse to children (hardcoded for your 17-joint hierarchy)
    if (joint_idx == J_PELVIS) {  // Root: Pelvis
        compute_fk(joints, J_SPINE, j->world_rot, j->world_pos);
        compute_fk(joints, J_L_HIP, j->world_rot, j->world_pos);
        compute_fk(joints, J_R_HIP, j->world_rot, j->world_pos);
    } else if (joint_idx == J_SPINE) {
        compute_fk(joints, J_THORAX, j->world_rot, j->world_pos);
    } else if (joint_idx == J_THORAX) {
        compute_fk(joints, J_NECK, j->world_rot, j->world_pos);
        compute_fk(joints, J_L_SHOULDER, j->world_rot, j->world_pos);
        compute_fk(joints, J_R_SHOULDER, j->world_rot, j->world_pos);
    } else if (joint_idx == J_NECK) {
        compute_fk(joints, J_HEAD, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_SHOULDER) {
        compute_fk(joints, J_L_ELBOW, j->world_rot, j->world_pos);
    } else if (joint_idx == J_L_ELBOW) {
        compute_fk(joints, J_L_WRIST, j->world_rot, j->world_pos);
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

void set_frames(Fighter* env){
    env->moveset[0].move_frame_start = 48;
    env->moveset[0].move_frame_end = env->moveset[0].move_frame_start + 12;
    env->moveset[0].total_root_distance = 0;
    env->moveset[0].motion_end_frame = env->moveset[0].move_frame_end;
    
    env->moveset[1].move_frame_start = 5;
    env->moveset[1].move_frame_end = env->moveset[1].move_frame_start + 29;
    env->moveset[1].total_root_distance = 0.6;
    env->moveset[1].motion_end_frame = env->moveset[1].move_frame_end - 14;

    env->moveset[2].move_frame_start = 16;
    env->moveset[2].move_frame_end = env->moveset[2].move_frame_start + 12 + 44;
    env->moveset[2].total_root_distance = 0.1;
    env->moveset[2].motion_end_frame = env->moveset[2].move_frame_end - 30;

    env->moveset[3].move_frame_start = 0;
    env->moveset[3].move_frame_end = env->moveset[3].move_frame_start + 40;
    env->moveset[3].total_root_distance = 0.7;
    env->moveset[3].motion_end_frame = env->moveset[3].move_frame_end - 20;

    env->moveset[4].move_frame_start = 5;
    env->moveset[4].move_frame_end = env->moveset[4].move_frame_start + 29;
    env->moveset[4].total_root_distance = 0.5;
    env->moveset[4].motion_end_frame = env->moveset[4].move_frame_end - 19;
}

void init(Fighter* env) {
    env->tick = 0;
    env->num_characters = 2;
    env->characters = (Character*)calloc(env->num_characters, sizeof(Character));
    load_moveset("resources/fighter/binaries/paul.bin", env);
    set_frames(env);
    for (int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        c->health = 100;
        c->pos_x = (i == 0) ? -2.0f : 5.0f; // spawn left/right
        c->pos_y = 1.0f;
        c->pos_z = 0.0f;
        c->facing = (i == 0) ? PI/2.0f : -PI/2.0f;
        c->state  = 0;
        c->num_shapes = 19;  
        c->shapes = calloc(c->num_shapes, sizeof(CollisionShape));
        c->push_shapes = calloc(8, sizeof(CollisionShape));
        c->num_joints = NUM_JOINTS;
        c->joints  = calloc(NUM_JOINTS, sizeof(Joint));
        c->bone_lengths = calloc(NUM_JOINTS, sizeof(float));
        init_skeleton(c); // joints + hierarchy
        store_bone_lengths(c);
        align_skeleton_to_mocap(c, &env->moveset[0]);
        init_shapes(c);   // collision capsules/spheres
        for(int j = 0; j < env->num_moves; j++){
            compute_local_rotations(&env->moveset[j], c);
        };
        
    }
    for(int i =0; i< env->num_characters; i++){
        int target = i == 0;
        Character *c = &env->characters[i];
        c->facing = atan2(env->characters[target].pos_z - c->pos_z, env->characters[target].pos_x - c->pos_x);
        Quat facing = quat_from_axis_angle((Vec3){0.0f, 1.0f, 0.0f}, c->facing);
        compute_fk(c->joints, J_PELVIS, facing, (Vec3){c->pos_x, c->pos_y, c->pos_z});
    }
    printf("init\n");
}

void c_reset(Fighter* env) {
    env->tick = 0;
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

void move_character(Fighter* env, int character_index, int target_index, int action) {
    Character* character = &env->characters[character_index];
    if (action == ACTION_LEFT) {
        character->pos_x -= 0.05 * cos(character->facing);
        character->pos_z -= 0.05 * sin(character->facing);
    } else if (action == ACTION_RIGHT) {
        character->pos_x += 0.05 * cos(character->facing);
        character->pos_z += 0.05 * sin(character->facing);
    } else if (action == ACTION_SIDESTEP_UP){
        sidestep(env, character, 1, target_index);
    } else if (action == ACTION_SIDESTEP_DOWN){
        sidestep(env, character, -1, target_index);
    } 
}

void replay_motion(Fighter* env, int frame, int move_idx, Character* c){
    int frame_count = env->moveset[move_idx].move_frame_count;
    frame = frame % frame_count;
    for(int i = 0; i < NUM_JOINTS; i++){
        if(i==J_HEAD) continue;
        //printf("data: %d frame_count: %d\n", i+frame*17, env->moveset[0].move_frame_count);
        float x = env->moveset[move_idx].joint_move_x[i+frame*17];
        float y = env->moveset[move_idx].joint_move_y[i+frame*17];
        float z = env->moveset[move_idx].joint_move_z[i+frame*17];
        //printf("x: %f, y: %f, z: %f i: %d\n", x,y,z,i);
        c->joints[i].world_pos = (Vec3){z,-y+1,x};
    }
}

void apply_move_frame(Character* c, Move* move, int frame) {
    frame = frame % move->move_frame_count;
    for (int j = 0; j < NUM_JOINTS; j++){
        c->joints[j].local_rot = move->local_rot[frame*NUM_JOINTS + j];
    }
    float distance_per_frame = move->total_root_distance / (move->motion_end_frame - move->move_frame_start);
    if(frame < move->motion_end_frame){

        float cos_facing = cosf(c->facing);
        float sin_facing = sinf(c->facing);
        
        c->pos_x += distance_per_frame * cos_facing;
        c->pos_z += distance_per_frame * sin_facing;
    }
}

int sphere_collision(CollisionShape* c1, Character* char1, int c1_act, CollisionShape* c2, Character* char2, int c2_act){
    if(c1->type!=3 || c2->type!=3) return 0;
    Vec3 p1 = char1->joints[c1->sphere.joint].world_pos;
    Vec3 p2 = char2->joints[c2->sphere.joint].world_pos;
    float r1 = c1->radius;
    float r2 = c2->radius;
    float rsum = r1+r2;

    Vec3 delta = vec3_sub(p2,p1);
    float dist2 = vec3_dot(delta,delta);
    if(dist2 >= rsum*rsum || dist2 < 1e-8f) return 0;
    // push back on characters
    float dist = sqrtf(dist2);
    float penetration = rsum - dist;
    if(penetration <= 0.001f)return 0;
    Character* mover = (c1_act == 0 && c2_act < 5 && c2_act>0) ? char2 : char1;
    Character* target = (c1_act == 0 && c2_act < 5 && c2_act>0) ? char1: char2;
    Vec3 forward = {
        sinf(mover->facing),
        0.0f,
        cosf(mover->facing)
    };
    forward = vec3_normalize(forward);
    // push
    float push_strength = 0.4f;
    target->pos_x += forward.x * penetration * push_strength;
    target->pos_z += forward.z * penetration * push_strength;

    mover->pos_x  += forward.x * -penetration * (1.0f - push_strength);
    mover->pos_z  += forward.z * -penetration * (1.0f - push_strength);

    return 1;

}


void c_step(Fighter* env) {
    for(int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        int action = env->actions[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;
        // movement
        if(action < 5){
            int target = i==0;
            move_character(env, i, target , action);
        }
    }

    // push collision resolution
    Character* c1 = &env->characters[0];
    Character* c2 = &env->characters[1];
    int c1_act = env->actions[0];
    int c2_act = env->actions[1];
    int collided = 0;
    for (int i = 0; i < 8 && !collided; i++) {
        for (int j = 0; j < 8; j++) {
            collided = sphere_collision(&c1->push_shapes[i], c1, c1_act,
                             &c2->push_shapes[j], c2, c2_act);
            if(collided){
                printf("colliding joints %d & %d\n", i,j);
                break;
            }
        }
    }
    for(int i = 0; i < env->num_characters; i++){
        Character* c = &env->characters[i];
        int action = env->actions[i];
        // fighting
        if(action >=5 || c->anim_timestep > 0){
            if(c->anim_timestep == 0){
                c->active_animation_idx = action - 5;
                c->anim_timestep = env->moveset[c->active_animation_idx].move_frame_start;
                printf("start frame: %d\n", c->anim_timestep);
            }
            Move* move = &env->moveset[c->active_animation_idx];
            apply_move_frame(c, move, c->anim_timestep);
            c->anim_timestep++;
            if(c->anim_timestep+1 == move->move_frame_end){
                c->anim_timestep =0;
                c->active_animation_idx=-1;
            }
        }
        //replay_motion(env,env->tick, 3, c);
        Quat facing = quat_from_axis_angle((Vec3){0.0f, -1.0f, 0.0f}, c->facing);
        Quat identity = {1,0,0,0};
        compute_fk(c->joints, J_PELVIS, facing, (Vec3){c->pos_x, c->pos_y, c->pos_z});

    }
    env->tick+=1;
}

typedef struct {
    int active_joint;
    int hover_axis;
    int drag_axis;
    Vector2 drag_start;
    Quat initial_rotation;
    float gizmo_scale;
    bool enabled;
} Gizmo;

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
    Gizmo* gizmo;
};


Gizmo* init_gizmo() {
    Gizmo* g = (Gizmo*)calloc(1, sizeof(Gizmo));
    g->active_joint = -1;
    g->hover_axis = GIZMO_AXIS_NONE;
    g->drag_axis = GIZMO_AXIS_NONE;
    g->gizmo_scale = 1.0f;
    g->enabled = true;
    return g;
}


// Check if ray intersects with a torus (rotation ring)
bool ray_intersects_torus(Ray ray, Vector3 center, Vector3 axis, float major_radius, float minor_radius) {
    // Simplified intersection test - check distance from ray to circle
    // This is an approximation that works well for thin tori
    
    // Project ray onto plane perpendicular to axis
    Vector3 n = Vector3Normalize(axis);
    float d = Vector3DotProduct(Vector3Subtract(center, ray.position), n) / Vector3DotProduct(ray.direction, n);
    
    if (d < 0) return false;
    
    Vector3 hit_point = Vector3Add(ray.position, Vector3Scale(ray.direction, d));
    Vector3 to_hit = Vector3Subtract(hit_point, center);
    
    // Remove component along axis
    to_hit = Vector3Subtract(to_hit, Vector3Scale(n, Vector3DotProduct(to_hit, n)));
    
    float dist_to_circle = fabsf(Vector3Length(to_hit) - major_radius);
    return dist_to_circle < minor_radius * 3.0f; // Generous hit zone
}

// Draw a rotation ring (torus) for the gizmo
void draw_rotation_ring(Vector3 center, Vector3 axis, float radius, Color color, bool highlighted) {
    int segments = 32;
    Vector3 perpendicular1, perpendicular2;
    
    // Find two vectors perpendicular to axis
    if (fabsf(axis.y) < 0.99f) {
        perpendicular1 = Vector3Normalize(Vector3CrossProduct(axis, (Vector3){0, 1, 0}));
    } else {
        perpendicular1 = Vector3Normalize(Vector3CrossProduct(axis, (Vector3){1, 0, 0}));
    }
    perpendicular2 = Vector3CrossProduct(axis, perpendicular1);
    
    // Draw the ring
    rlPushMatrix();
    rlTranslatef(center.x, center.y, center.z);
    
    if (highlighted) {
        rlSetLineWidth(4.0f);
    } else {
        rlSetLineWidth(2.0f);
    }
    
    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, highlighted ? 255 : 200);
    
    for (int i = 0; i < segments; i++) {
        float angle1 = (float)i * 2.0f * PI / segments;
        float angle2 = (float)((i + 1) % segments) * 2.0f * PI / segments;
        
        Vector3 p1 = Vector3Add(
            Vector3Scale(perpendicular1, cosf(angle1) * radius),
            Vector3Scale(perpendicular2, sinf(angle1) * radius)
        );
        Vector3 p2 = Vector3Add(
            Vector3Scale(perpendicular1, cosf(angle2) * radius),
            Vector3Scale(perpendicular2, sinf(angle2) * radius)
        );
        
        rlVertex3f(p1.x, p1.y, p1.z);
        rlVertex3f(p2.x, p2.y, p2.z);
    }
    
    rlEnd();
    rlPopMatrix();
}

// Update gizmo interaction
void update_gizmo(Gizmo* g, Character* character, Camera3D* camera) {
    if (!g->enabled || g->active_joint < 0) return;
    
    Joint* joint = &character->joints[g->active_joint];
    Vector3 joint_pos = (Vector3){joint->world_pos.x, joint->world_pos.y, joint->world_pos.z};
    
    Ray mouse_ray = GetMouseRay(GetMousePosition(), *camera);
    float gizmo_radius = g->gizmo_scale * 0.5f;
    float ring_thickness = 0.02f;
    
    // Check for hover if not dragging
    if (g->drag_axis == GIZMO_AXIS_NONE) {
        g->hover_axis = GIZMO_AXIS_NONE;
        
        // Check X axis (red)
        if (ray_intersects_torus(mouse_ray, joint_pos, (Vector3){1, 0, 0}, gizmo_radius, ring_thickness)) {
            g->hover_axis = GIZMO_AXIS_X;
        }
        // Check Y axis (green)
        else if (ray_intersects_torus(mouse_ray, joint_pos, (Vector3){0, 1, 0}, gizmo_radius, ring_thickness)) {
            g->hover_axis = GIZMO_AXIS_Y;
        }
        // Check Z axis (blue)
        else if (ray_intersects_torus(mouse_ray, joint_pos, (Vector3){0, 0, 1}, gizmo_radius, ring_thickness)) {
            g->hover_axis = GIZMO_AXIS_Z;
        }
        
        // Start drag on mouse down
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && g->hover_axis != GIZMO_AXIS_NONE) {
            g->drag_axis = g->hover_axis;
            g->drag_start = GetMousePosition();
            g->initial_rotation = joint->local_rot;
        }
    }
    
    // Handle dragging
    if (g->drag_axis != GIZMO_AXIS_NONE) {
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            g->drag_axis = GIZMO_AXIS_NONE;
        } else {
            Vector2 mouse_delta = Vector2Subtract(GetMousePosition(), g->drag_start);
            float rotation_speed = 0.01f;
            float angle = (mouse_delta.x - mouse_delta.y) * rotation_speed; // Use both x and y for rotation
            
            Vec3 axis;
            switch (g->drag_axis) {
                case GIZMO_AXIS_X: axis = (Vec3){1, 0, 0}; break;
                case GIZMO_AXIS_Y: axis = (Vec3){0, 1, 0}; break;
                case GIZMO_AXIS_Z: axis = (Vec3){0, 0, 1}; break;
            }
            
            // Apply rotation as delta from initial
            Quat delta_rot = quat_from_axis_angle(axis, angle);
            joint->local_rot = quat_mul(g->initial_rotation, delta_rot);
        }
    }
}

// Draw the gizmo
void draw_gizmo(Gizmo* g, Character* character, Camera3D* camera) {
    if (!g->enabled || g->active_joint < 0) return;
    
    Joint* joint = &character->joints[g->active_joint];
    Vector3 joint_pos = (Vector3){joint->world_pos.x, joint->world_pos.y, joint->world_pos.z};
    
    float radius = g->gizmo_scale * 0.5f;
    
    // Draw rotation rings
    draw_rotation_ring(joint_pos, (Vector3){1, 0, 0}, radius, RED, g->hover_axis == GIZMO_AXIS_X || g->drag_axis == GIZMO_AXIS_X);
    draw_rotation_ring(joint_pos, (Vector3){0, 1, 0}, radius, GREEN, g->hover_axis == GIZMO_AXIS_Y || g->drag_axis == GIZMO_AXIS_Y);
    draw_rotation_ring(joint_pos, (Vector3){0, 0, 1}, radius, BLUE, g->hover_axis == GIZMO_AXIS_Z || g->drag_axis == GIZMO_AXIS_Z);
    
    // Draw center sphere
    DrawSphere(joint_pos, 0.05f, WHITE);
}

Client* make_client(Fighter* env){    
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "pufferlib fighter");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/puffers_128.png");
    
    client->default_camera_position = (Vector3){ 
        -1.0f,           // same x as target
        4.0f,   // 20 units above target
        0.0f    // 20 units behind target
    };

    client->default_camera_target = (Vector3){-5, 2, 0};
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };  // y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    client->gizmo = init_gizmo();
    client->gizmo->active_joint = J_L_ELBOW;
    DisableCursor(); 
    return client;
}

void UpdateCameraFighter(Fighter* env, Client* client){
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
}

void DrawWireSphere(Vector3 center, float radius, int rings, int slices, Color color) {
    for (int i = 0; i <= rings; i++) {
        float lat = PI * ((float)i / (float)rings - 0.5f);  // from -PI/2 to PI/2
        float y = sinf(lat);
        float r = cosf(lat);
        Vector3 prev = {0};
        for (int j = 0; j <= slices; j++) {
            float lon = 2.0f * PI * (float)j / (float)slices;
            float x = cosf(lon) * r;
            float z = sinf(lon) * r;
            Vector3 curr = {
                center.x + radius * x,
                center.y + radius * y,
                center.z + radius * z
            };
            if (j > 0) DrawLine3D(prev, curr, color);
            prev = curr;
        }
    }

    // Draw longitude lines (vertical)
    for (int j = 0; j <= slices; j++) {
        float lon = 2.0f * PI * (float)j / (float)slices;
        Vector3 prev = {0};
        for (int i = 0; i <= rings; i++) {
            float lat = PI * ((float)i / (float)rings - 0.5f);
            float y = sinf(lat);
            float r = cosf(lat);
            float x = cosf(lon) * r;
            float z = sinf(lon) * r;
            Vector3 curr = {
                center.x + radius * x,
                center.y + radius * y,
                center.z + radius * z
            };
            if (i > 0) DrawLine3D(prev, curr, color);
            prev = curr;
        }
    }
}

void c_render(Fighter* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    if (IsKeyPressed(KEY_ONE)) client->gizmo->active_joint = J_L_ELBOW;
    if (IsKeyPressed(KEY_TWO)) client->gizmo->active_joint = J_R_ELBOW;
    if (IsKeyPressed(KEY_THREE)) client->gizmo->active_joint = J_L_SHOULDER;
    if (IsKeyPressed(KEY_FOUR)) client->gizmo->active_joint = J_R_SHOULDER;
    if (IsKeyPressed(KEY_FIVE)) client->gizmo->active_joint = J_L_WRIST;
    if (IsKeyPressed(KEY_SIX)) client->gizmo->active_joint = J_R_WRIST;
    if (IsKeyPressed(KEY_SEVEN)) client->gizmo->active_joint = J_SPINE;
    if (IsKeyPressed(KEY_EIGHT)) client->gizmo->active_joint = J_THORAX;

    update_gizmo(client->gizmo, &env->characters[0], &client->camera);
    UpdateCameraFighter(env, client);
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
        float arrow_length = 2.0f;
        float arrow_head_size = 0.3f;
        
        // Starting position
        Vector3 start = { chara->pos_x, chara->pos_y, chara->pos_z };
        DrawSphere(start, 0.20f, fighter_color);
        // Calculate arrow endpoint (assuming facing is rotation around Y-axis)
        Vector3 arrow_end = {
            chara->pos_x + cosf(chara->facing) * arrow_length,
            chara->pos_y,  // Keep same height
            chara->pos_z + sinf(chara->facing) * arrow_length
        };
        
        // Draw main arrow line
        DrawLine3D(start, arrow_end, fighter_color);
        
        // Draw arrowhead
        float head_angle1 = chara->facing + 2.8f;
        float head_angle2 = chara->facing - 2.8f;
        
        Vector3 head1 = {
            arrow_end.x + cosf(head_angle1) * arrow_head_size,
            arrow_end.y,
            arrow_end.z + sinf(head_angle1) * arrow_head_size
        };
        Vector3 head2 = {
            arrow_end.x + cosf(head_angle2) * arrow_head_size,
            arrow_end.y,
            arrow_end.z + sinf(head_angle2) * arrow_head_size
        };
        
        DrawLine3D(arrow_end, head1, fighter_color);
        DrawLine3D(arrow_end, head2, fighter_color);
        
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
        for(int i = 0; i < 8; i++){
            CollisionShape *s = &chara->push_shapes[i];
            int j = s->sphere.joint;
            Vec3 J_base = chara->joints[j].world_pos;

            Quat joint_rot = chara->joints[j].world_rot;
            Vec3 rotated_offset = quat_rotate_vec(joint_rot, s->sphere.local_offset);
            Vector3 J = (Vector3){J_base.x + rotated_offset.x, J_base.y + rotated_offset.y, J_base.z + rotated_offset.z};

            DrawWireSphere(J, s->radius, 5, 24, fighter_color);
        }
    }
    //draw_gizmo(client->gizmo, &env->characters[0], &client->camera);
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
    float elbow_x = env->characters[0].joints[8].local_rot.x;
    float elbow_y = env->characters[0].joints[8].local_rot.y;
    float elbow_z = env->characters[0].joints[8].local_rot.z;
    float elbow_w = env->characters[0].joints[8].local_rot.w;

    DrawText(TextFormat("elbow quat w:%f x:%f y:%f z:%f\n", elbow_w, elbow_x, elbow_y, elbow_z), char1_x + 10, health_bar_y + 50, 20, WHITE);    
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
