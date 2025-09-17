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
#define NUM_JOINTS 14
#define J_TORSO      0
#define J_HEAD       1
#define J_SHOULDER_R 2
#define J_ELBOW_R    3
#define J_WRIST_R    4
#define J_SHOULDER_L 5
#define J_ELBOW_L    6
#define J_WRIST_L    7
#define J_HIP_R      8
#define J_KNEE_R     9
#define J_ANKLE_R   10
#define J_HIP_L    11
#define J_KNEE_L   12
#define J_ANKLE_L  13

// only use floats!
typedef struct {
    float score;
    float n; // required as the last field 
} Log;


typedef struct {
    int type;
    float radius;

    union {
        // Capsule defined by two joints (limbs: shoulder→elbow, elbow→wrist, hip→knee, etc.)
        struct { int joint_a, joint_b; float pad_a, pad_b; } jnt;

        // Capsule defined by one joint + a local offset (useful for torso/head)
        struct { int parent; float off_x, off_y, off_z; float height; } off;

        // Simple sphere at a single joint (good for fists, feet, head hitboxes)
        struct { int joint; } sph;
    };
} CollisionShape;


typedef struct { 
    // ---- root pose ----
    float pos_x, pos_y, pos_z;   // translation
    float facing;                // yaw (whole-body facing)

    // ---- joints ----
    int   num_joints;
    int  *parent;      // parent index per joint (-1 for root)
    float *length;     // bone length
    float *pitch;      // joint pitch angle
    float *yaw;        // joint yaw angle
    float *roll;       // optional

    // FK outputs
    float *world_x;
    float *world_y;
    float *world_z;

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

/*void init_collision_shapes(CollisionShape* hurt_shapes, CollisionShape* hit_shapes, float facing) {
    hurt_shapes[0].type = shape_capsule;
    hurt_shapes[0].offset_x = 0.0f;
    hurt_shapes[0].offset_y = 1.6f;  // center at mid-body (half height)
    hurt_shapes[0].offset_z = 0.0f;
    hurt_shapes[0].height = 0.5f;    // character height
    hurt_shapes[0].radius = 0.2f;    // body width

    // left arm: horizontal capsule, extends based on facing angle
    hurt_shapes[1].type = shape_capsule;
    hurt_shapes[1].offset_x = 0.0f;
    hurt_shapes[1].offset_y = 1.8f;  // shoulder height
    hurt_shapes[1].offset_z = -0.25f;
    hurt_shapes[1].rotation_offset_x = 0.0f;
    hurt_shapes[1].rotation_offset_z = 0.0f;
    hurt_shapes[1].height = 0.35f;    // arm length (as capsule "height" along axis)
    hurt_shapes[1].radius = 0.075f;   // arm thickness

    // right arm: similar, opposite offset
    hurt_shapes[2].type = shape_capsule;
    hurt_shapes[2].offset_x = 0.0f;
    hurt_shapes[2].offset_y = 1.8f;
    hurt_shapes[2].offset_z = 0.25f;
    hurt_shapes[2].rotation_offset_x = 0.0f;
    hurt_shapes[2].rotation_offset_z = 0.0f;
    hurt_shapes[2].height = 0.35f;
    hurt_shapes[2].radius = 0.075f;

    // left leg: single vertical capsule for both (or split for precision)
    hurt_shapes[3].type = shape_capsule;
    hurt_shapes[3].offset_x = 0.0f;
    hurt_shapes[3].offset_y = 1.35f;  // from ground to hips
    hurt_shapes[3].offset_z = -0.15f;
    hurt_shapes[3].rotation_offset_x = 0.0f;
    hurt_shapes[3].rotation_offset_z = 0.0f;
    hurt_shapes[3].height = 0.5f;    // leg height
    hurt_shapes[3].radius = 0.1f;   // leg width 
    // right leg
    hurt_shapes[4].type = shape_capsule;
    hurt_shapes[4].offset_x = 0.0f;
    hurt_shapes[4].offset_y = 1.35f;  // from ground to hips
    hurt_shapes[4].offset_z = 0.15f;
    hurt_shapes[4].rotation_offset_x = 0.0f;
    hurt_shapes[4].rotation_offset_z = 0.0f;
    hurt_shapes[4].height = 0.5f;    // leg height
    hurt_shapes[4].radius = 0.1f;   // leg width

    // head
    hurt_shapes[5].type = shape_capsule;
    hurt_shapes[5].offset_x = 0.0f;
    hurt_shapes[5].offset_y = 2.2f;
    hurt_shapes[5].offset_z = 0.0f;
    hurt_shapes[5].height = 0.2f;    // head height
    hurt_shapes[5].radius = 0.2f;    // head width

    // left forearm
    hurt_shapes[6].type = shape_capsule;
    hurt_shapes[6].offset_x = 0.1f;
    hurt_shapes[6].offset_y = 0.0f;   // elbow level
    hurt_shapes[6].offset_z = 0.0f; // forward from torso
    hurt_shapes[6].height = 0.3f;     // forearm length
    hurt_shapes[6].radius = 0.07f;

    // right forearm
    hurt_shapes[7].type = shape_capsule;
    hurt_shapes[7].offset_x = 0.1f;
    hurt_shapes[7].offset_y = 0.0f;
    hurt_shapes[7].offset_z = 0.0f;
    hurt_shapes[7].height = 0.3f;
    hurt_shapes[7].radius = 0.07f;

    // left shin
    hurt_shapes[8].type = shape_capsule;
    hurt_shapes[8].offset_x = 0.0f;
    hurt_shapes[8].offset_y = 0.0f;   // knee level
    hurt_shapes[8].offset_z = -0.15f;
    hurt_shapes[8].height = 0.5f;     // shin length
    hurt_shapes[8].radius = 0.09f;

    // right shin
    hurt_shapes[9].type = shape_capsule;
    hurt_shapes[9].offset_x = 0.0f;
    hurt_shapes[9].offset_y = 0.0f;
    hurt_shapes[9].offset_z = 0.15f;
    hurt_shapes[9].height = 0.5f;
    hurt_shapes[9].radius = 0.09f;
    // for other shapes (hit_shapes), set to invalid until a move activates them
    for (int i = 0; i < 6; i++) {
        hit_shapes[i].type = shape_none;  // add shape_none to enum if needed
    }
}



void init_character(Character* character, float type) {
    character->health = 100;
    if (type == 0){
        character->pos_x = -5;
    } else {
        character->pos_x = 5;
    }
    character->pos_y = 0;
    character->pos_z = 0;
    character->vel_x = 0;
    character->vel_y = 0;
    character->vel_z = 0;
    character->facing = type;
    character->current_move = 0;
    character->current_action_frames = 0;
    character->animation_frame = 0;
    character->frame_advantage = 0;
    character->state = state_idle;
    character->hurt_shapes = (CollisionShape*)calloc(10, sizeof(CollisionShape));
    character->hit_shapes = (CollisionShape*)calloc(6, sizeof(CollisionShape));
    init_collision_shapes(character->hurt_shapes, character->hit_shapes, character->facing);
}
*/
void init_skeleton(Character* c){
    c->num_joints = NUM_JOINTS;
    c->parent  = calloc(NUM_JOINTS, sizeof(int));
    c->length  = calloc(NUM_JOINTS, sizeof(float));
    c->pitch   = calloc(NUM_JOINTS, sizeof(float));
    c->yaw     = calloc(NUM_JOINTS, sizeof(float));
    c->roll    = calloc(NUM_JOINTS, sizeof(float));
    c->world_x = calloc(NUM_JOINTS, sizeof(float));
    c->world_y = calloc(NUM_JOINTS, sizeof(float));
    c->world_z = calloc(NUM_JOINTS, sizeof(float));

    // skeleton relationships
    c->parent[J_TORSO]      = -1;        // root of skeleton
    c->parent[J_HEAD]       = J_TORSO;

    c->parent[J_SHOULDER_R] = J_TORSO;
    c->parent[J_ELBOW_R]    = J_SHOULDER_R;
    c->parent[J_WRIST_R]    = J_ELBOW_R;

    c->parent[J_SHOULDER_L] = J_TORSO;
    c->parent[J_ELBOW_L]    = J_SHOULDER_L;
    c->parent[J_WRIST_L]    = J_ELBOW_L;

    c->parent[J_HIP_R]      = J_TORSO;
    c->parent[J_KNEE_R]     = J_HIP_R;
    c->parent[J_ANKLE_R]    = J_KNEE_R;

    c->parent[J_HIP_L]      = J_TORSO;
    c->parent[J_KNEE_L]     = J_HIP_L;
    c->parent[J_ANKLE_L]    = J_KNEE_L;

    // Bone lengths (approximate, tweak as needed)
    c->length[J_HEAD]       = 0.7f;  // torso→head
    c->length[J_SHOULDER_R] = 0.35f; // upper arm
    c->length[J_ELBOW_R]    = 0.5f; // forearm
    c->length[J_WRIST_R]    = 0.1f;
    c->length[J_SHOULDER_L] = 0.35f;
    c->length[J_ELBOW_L]    = 0.5f;
    c->length[J_WRIST_L]    = 0.1f;
    c->length[J_HIP_R]      = 0.5f;  // thigh
    c->length[J_KNEE_R]     = 0.5f;  // shin
    c->length[J_ANKLE_R]    = 0.1f;
    c->length[J_HIP_L]      = 0.5f;
    c->length[J_KNEE_L]     = 0.5f;
    c->length[J_ANKLE_L]    = 0.1f;

    // Init joint angles to neutral pose
    for (int i = 0; i < NUM_JOINTS; i++) {
        c->pitch[i] = 0.0f;
        c->yaw[i]   = 0.0f;
        c->roll[i]  = 0.0f;
    }
}

void init_shapes(Character *c) {
    c->num_shapes = 14;  
    c->shapes = calloc(c->num_shapes, sizeof(CollisionShape));

    // Torso
    c->shapes[0].type = SHAPE_CAPSULE_OFFSET;
    c->shapes[0].radius = 0.20f;
    c->shapes[0].off.parent = J_TORSO;
    c->shapes[0].off.off_x = 0.0f;
    c->shapes[0].off.off_y = 0.5f;
    c->shapes[0].off.off_z = 0.0f;
    c->shapes[0].off.height = 0.75f;

    // Head
    c->shapes[1].type = SHAPE_SPHERE_JOINT;
    c->shapes[1].radius = 0.20f;
    c->shapes[1].sph.joint = J_HEAD;

    // Right arm
    c->shapes[2].type = SHAPE_CAPSULE_JOINT;  // upper arm
    c->shapes[2].radius = 0.075f;
    c->shapes[2].jnt.joint_a = J_SHOULDER_R;
    c->shapes[2].jnt.joint_b = J_ELBOW_R;

    c->shapes[3].type = SHAPE_CAPSULE_JOINT;  // forearm
    c->shapes[3].radius = 0.07f;
    c->shapes[3].jnt.joint_a = J_ELBOW_R;
    c->shapes[3].jnt.joint_b = J_WRIST_R;

    c->shapes[4].type = SHAPE_SPHERE_JOINT;   // wrist sphere
    c->shapes[4].radius = 0.10f;
    c->shapes[4].sph.joint = J_WRIST_R;

    // Left arm
    c->shapes[5].type = SHAPE_CAPSULE_JOINT;
    c->shapes[5].radius = 0.075f;
    c->shapes[5].jnt.joint_a = J_SHOULDER_L;
    c->shapes[5].jnt.joint_b = J_ELBOW_L;

    c->shapes[6].type = SHAPE_CAPSULE_JOINT;
    c->shapes[6].radius = 0.07f;
    c->shapes[6].jnt.joint_a = J_ELBOW_L;
    c->shapes[6].jnt.joint_b = J_WRIST_L;

    c->shapes[7].type = SHAPE_SPHERE_JOINT;
    c->shapes[7].radius = 0.10f;
    c->shapes[7].sph.joint = J_WRIST_L;

    // Right leg
    c->shapes[8].type = SHAPE_CAPSULE_JOINT;  // thigh
    c->shapes[8].radius = 0.10f;
    c->shapes[8].jnt.joint_a = J_HIP_R;
    c->shapes[8].jnt.joint_b = J_KNEE_R;

    c->shapes[9].type = SHAPE_CAPSULE_JOINT;  // shin
    c->shapes[9].radius = 0.09f;
    c->shapes[9].jnt.joint_a = J_KNEE_R;
    c->shapes[9].jnt.joint_b = J_ANKLE_R;

    c->shapes[10].type = SHAPE_SPHERE_JOINT;  // ankle sphere
    c->shapes[10].radius = 0.10f;
    c->shapes[10].sph.joint = J_ANKLE_R;

    // Left leg
    c->shapes[11].type = SHAPE_CAPSULE_JOINT;
    c->shapes[11].radius = 0.10f;
    c->shapes[11].jnt.joint_a = J_HIP_L;
    c->shapes[11].jnt.joint_b = J_KNEE_L;

    c->shapes[12].type = SHAPE_CAPSULE_JOINT;
    c->shapes[12].radius = 0.09f;
    c->shapes[12].jnt.joint_a = J_KNEE_L;
    c->shapes[12].jnt.joint_b = J_ANKLE_L;

    c->shapes[13].type = SHAPE_SPHERE_JOINT;
    c->shapes[13].radius = 0.10f;
    c->shapes[13].sph.joint = J_ANKLE_L;
}



void init(Fighter* env) {
    env->num_characters = 2;
    env->characters = (Character*)calloc(env->num_characters, sizeof(Character));
    /*init_character(&env->characters[0], 0);
    init_character(&env->characters[1], pi);*/

    
    for (int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        c->health = 100;
        c->pos_x = (i == 0) ? -5.0f : 5.0f; // spawn left/right
        c->pos_y = 0.0f;
        c->pos_z = 0.0f;
        c->facing = (i == 0) ? 0.0f : PI;   // face each other
        c->state  = 0;

        init_skeleton(c); // joints + hierarchy
        init_shapes(c);   // collision capsules/spheres
    }
    printf("init\n");
}

void c_reset(Fighter* env) {
    /*init_collision_shapes(env->characters[0].hurt_shapes, env->characters[0].hit_shapes, 1);
    init_collision_shapes(env->characters[1].hurt_shapes, env->characters[1].hit_shapes, -1);
    */
    for (int i = 0; i < env->num_characters; i++) {
        Character *c = &env->characters[i];
        c->health = 100;
        c->pos_x = (i == 0) ? -5.0f : 5.0f; // spawn left/right
        c->pos_y = 0.0f;
        c->pos_z = 0.0f;
        c->facing = (i == 0) ? 0.0f : PI;   // face each other
        c->state  = 0;

        init_skeleton(c); // joints + hierarchy
        init_shapes(c);   // collision capsules/spheres
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

void adjust_skeleton(Fighter* env, int character_index) {
    Character* character = &env->characters[character_index];
    for (int i = 0; i < 10; i++) {   
        if(i == 1 || i == 2 || i == 3 || i == 4 || i == 6 || i==7 || i==8 || i==9){  // fixed the condition
            CollisionShape* shape = &character->hurt_shapes[i];
            // store rotated positions in rotation_offset fields, preserve original offsets
            shape->rotation_offset_x = shape->offset_x * cos(character->facing) - shape->offset_z * sin(character->facing);
            shape->rotation_offset_z = shape->offset_x * sin(character->facing) + shape->offset_z * cos(character->facing);
        }
    }
}*/
void update_fk(Character *c) {
    // Root at character position
    c->world_x[J_TORSO] = c->pos_x;
    c->world_y[J_TORSO] = c->pos_y + 2.2f; // torso at chest height
    c->world_z[J_TORSO] = c->pos_z;

    // Head above torso
    c->world_x[J_HEAD] = c->world_x[J_TORSO];
    c->world_y[J_HEAD] = c->world_y[J_TORSO] + c->length[J_HEAD];
    c->world_z[J_HEAD] = c->world_z[J_TORSO];
    
    float shoulder_width = 0.25f;
    float shoulder_height = 0.25f;
    // Right arm (straight out)
    c->world_x[J_SHOULDER_R] = c->world_x[J_TORSO] + shoulder_width;
    c->world_y[J_SHOULDER_R] = c->world_y[J_TORSO] + shoulder_height;
    c->world_z[J_SHOULDER_R] = c->world_z[J_TORSO];

    c->world_x[J_ELBOW_R] = c->world_x[J_SHOULDER_R] + c->length[J_SHOULDER_R];
    c->world_y[J_ELBOW_R] = c->world_y[J_SHOULDER_R];
    c->world_z[J_ELBOW_R] = c->world_z[J_SHOULDER_R];

    c->world_x[J_WRIST_R] = c->world_x[J_ELBOW_R] + c->length[J_ELBOW_R];
    c->world_y[J_WRIST_R] = c->world_y[J_ELBOW_R];
    c->world_z[J_WRIST_R] = c->world_z[J_ELBOW_R];

    // Left arm (mirror)
    c->world_x[J_SHOULDER_L] = c->world_x[J_TORSO] - shoulder_width;
    c->world_y[J_SHOULDER_L] = c->world_y[J_TORSO] + shoulder_height;
    c->world_z[J_SHOULDER_L] = c->world_z[J_TORSO];

    c->world_x[J_ELBOW_L] = c->world_x[J_SHOULDER_L] - c->length[J_SHOULDER_L];
    c->world_y[J_ELBOW_L] = c->world_y[J_SHOULDER_L];
    c->world_z[J_ELBOW_L] = c->world_z[J_SHOULDER_L];

    c->world_x[J_WRIST_L] = c->world_x[J_ELBOW_L] - c->length[J_ELBOW_L];
    c->world_y[J_WRIST_L] = c->world_y[J_ELBOW_L];
    c->world_z[J_WRIST_L] = c->world_z[J_ELBOW_L];

    // Hips
    float hip_spacing = 0.2f;
    c->world_x[J_HIP_R] = c->world_x[J_TORSO] + hip_spacing;
    c->world_y[J_HIP_R] = c->world_y[J_TORSO] - 0.5f;
    c->world_z[J_HIP_R] = c->world_z[J_TORSO];

    c->world_x[J_HIP_L] = c->world_x[J_TORSO] - hip_spacing;
    c->world_y[J_HIP_L] = c->world_y[J_TORSO] - 0.5f;
    c->world_z[J_HIP_L] = c->world_z[J_TORSO];

    // Right leg down
    c->world_x[J_KNEE_R] = c->world_x[J_HIP_R];
    c->world_y[J_KNEE_R] = c->world_y[J_HIP_R] - c->length[J_HIP_R];
    c->world_z[J_KNEE_R] = c->world_z[J_HIP_R];

    c->world_x[J_ANKLE_R] = c->world_x[J_KNEE_R];
    c->world_y[J_ANKLE_R] = c->world_y[J_KNEE_R] - c->length[J_KNEE_R];
    c->world_z[J_ANKLE_R] = c->world_z[J_KNEE_R];

    // Left leg down
    c->world_x[J_KNEE_L] = c->world_x[J_HIP_L];
    c->world_y[J_KNEE_L] = c->world_y[J_HIP_L] - c->length[J_HIP_L];
    c->world_z[J_KNEE_L] = c->world_z[J_HIP_L];

    c->world_x[J_ANKLE_L] = c->world_x[J_KNEE_L];
    c->world_y[J_ANKLE_L] = c->world_y[J_KNEE_L] - c->length[J_KNEE_L];
    c->world_z[J_ANKLE_L] = c->world_z[J_KNEE_L];
}

void c_step(Fighter* env) {
    for(int i = 0; i < env->num_characters; i++) {
        env->rewards[i] = 0;
        env->terminals[i] = 0;
        update_fk(&env->characters[i]);
        //move_character(env, i, (i + 1) % env->num_characters, env->actions[i]);
        //adjust_skeleton(env, i);
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
    /*for (int c = 0; c < env->num_characters; c++) {
        character* chara = &env->characters[c];
        color fighter_color = (c == 0) ? red : blue;  // differentiate players
        
        for (int i = 0; i < 10; i++) {  // assuming 4 hurt_shapes
            collisionshape* shape = &chara->hurt_shapes[i];
            if (shape->type != shape_capsule) continue;

            // compute absolute positions, adjusted by character's pos and facing
            vector3 center = (vector3){
                chara->pos_x + shape->rotation_offset_x,
                chara->pos_y + shape->offset_y,
                chara->pos_z + shape->rotation_offset_z
            };
            //drawsphere(center, shape->radius+0.01,  black);

            // determine start and end points based on orientation
            // assume: torso/legs vertical (along y), arms horizontal (along x)
            vector3 start_pos, end_pos;
            if (i == 0) {  // vertical (torso/legs): start at bottom, end at top
                start_pos = (vector3){ center.x, center.y - (shape->height / 2.0f), center.z };
                end_pos = (vector3){ center.x, center.y + (shape->height / 2.0f), center.z };
            } else if (i==1 || i==2){
                // arms: use pre-computed rotation offsets from adjust_skeleton
                start_pos = (vector3){ 
                    center.x,
                    center.y, 
                    center.z
                };
                
                // arm extends outward from shoulder in the facing direction
                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                end_pos = (vector3){ 
                    start_pos.x + (shape->height) * facing_x, 
                    center.y, 
                    start_pos.z + (shape->height) * facing_z 
                };
            } else if (i==3 || i==4){
                // legs: hips rotate with body, feet stay planted
                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                if (i == 3) {  // left leg
                    vector3 hip_pos = (vector3){
                        center.x,
                        center.y,  // top of leg (hip level)
                        center.z
                    };
                    
                    vector3 knee_pos = (vector3){
                        hip_pos.x,  // original offset, no rotation
                        center.y - shape->height,  // bottom of leg (foot level)
                        hip_pos.z   // original offset, no rotation
                    };
                    printf("knee_pos y: %f\n", knee_pos.y);
                    start_pos = hip_pos;   // bottom of leg (foot)
                    end_pos = knee_pos;      // top of leg (hip)
                } else {  // right leg
                    vector3 hip_pos = (vector3){
                        center.x,
                        center.y,  // top of leg (hip level)
                        center.z
                    };
                    vector3 knee_pos = (vector3){
                        hip_pos.x,  // original offset, no rotation
                        center.y - shape->height,  // bottom of leg (foot level)
                        hip_pos.z   // original offset, no rotation
                    };
                    start_pos = hip_pos;   // bottom of leg (foot)
                    end_pos = knee_pos;      // top of leg (hip)
                }
            }
            else if (i == 6 | i == 7){
                collisionshape* left_arm = &chara->hurt_shapes[1];
                collisionshape* right_arm = &chara->hurt_shapes[2];
                
                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                collisionshape* arm = (i==6) ? left_arm : right_arm;
                start_pos = (vector3){
                    center.x + arm->rotation_offset_x + (arm->height) * facing_x,
                    center.y + arm->offset_y, 
                    center.z + arm->rotation_offset_z + (arm->height) * facing_z
                };

                end_pos = (vector3){ 
                    start_pos.x + (shape->height) * facing_x, 
                    chara->pos_y + arm->offset_y, 
                    start_pos.z + (shape->height) * facing_z 
                };
            }
            else if (i==8 || i==9){
                collisionshape* left_thigh = &chara->hurt_shapes[3];
                collisionshape* right_thigh = &chara->hurt_shapes[4];

                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                collisionshape* leg = (i==8)? left_thigh : right_thigh;
                vector3 knee_pos = (vector3){
                    chara->pos_x + leg->rotation_offset_x,
                    center.y + leg->offset_y - (leg->height),
                    chara->pos_z + leg->rotation_offset_z
                };

                vector3 foot_pos = (vector3){
                    knee_pos.x,
                    knee_pos.y - (shape->height),
                    knee_pos.z,
                };

                start_pos = knee_pos;
                end_pos = foot_pos;
                
            }
            else {
                // head: stays centered
                start_pos = (vector3){ center.x, center.y, center.z };
                end_pos = (vector3){ center.x, center.y, center.z };
            }
            if(i ==1) fighter_color = pink;
            if(i ==2) fighter_color = green;
            if(i ==3) fighter_color = yellow;
            if(i ==4) fighter_color = purple;
            if(i ==6) fighter_color = red;
            if(i ==7) fighter_color = blue;
            if(i ==8) fighter_color = orange;
            if(i ==9) fighter_color = brown;
            // draw the capsule (solid for body, or use drawcapsulewires for outline)
            drawcapsule(start_pos, end_pos, shape->radius, 16, 8, fighter_color);  // 16 slices, 8 rings for smoothness
            drawcapsulewires(start_pos, end_pos, shape->radius, 16, 8, black);  // wireframe for debug
        }
    }
    */
    for (int ci = 0; ci < env->num_characters; ci++) {
        Character *chara = &env->characters[ci];
        Color fighter_color = (ci == 0) ? RED : BLUE;

        for (int i = 0; i < chara->num_shapes; i++) {
            CollisionShape *s = &chara->shapes[i];

            if (s->type == SHAPE_CAPSULE_JOINT) {
                int a = s->jnt.joint_a;
                int b = s->jnt.joint_b;

                Vector3 A = { chara->world_x[a], chara->world_y[a], chara->world_z[a] };
                Vector3 B = { chara->world_x[b], chara->world_y[b], chara->world_z[b] };
                Vector3 dir = Vector3Normalize(Vector3Subtract(B, A));

                DrawCapsule(A, B, s->radius, 16, 8, fighter_color);
                DrawCapsuleWires(A, B, s->radius, 16, 8, BLACK);
            }

            else if (s->type == SHAPE_CAPSULE_OFFSET) {
                int p = s->off.parent;
                Vector3 P = { chara->world_x[p], chara->world_y[p], chara->world_z[p] };

                Vector3 A = { P.x, P.y - (s->off.height / 2.0f), P.z };
                Vector3 B = { P.x, P.y + (s->off.height / 2.0f), P.z };

                DrawCapsule(A, B, s->radius, 16, 8, fighter_color);
                DrawCapsuleWires(A, B, s->radius, 16, 8, BLACK);
            }

            else if (s->type == SHAPE_SPHERE_JOINT) {
                int j = s->sph.joint;
                Vector3 J = { chara->world_x[j], chara->world_y[j], chara->world_z[j] };
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
