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
#define STATE_IDLE 0 
#define STATE_WALKING 1
#define STATE_SIDESTEPPING 2
#define STATE_CROUCHING 3
#define STATE_JUMPING 4
#define STATE_ATTACKING 5
#define STATE_BLOCKING 6
#define STATE_HITSTUN 7
#define STATE_KNOCKDOWN 8
#define STATE_RECOVERING 9

// collision shapes
#define SHAPE_CAPSULE 0
#define SHAPE_BOX 1
#define SHAPE_NONE 2
typedef struct Client Client;
// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field 
} Log;

typedef struct {
    int type;
    float offset_x;
    float offset_y;
    float offset_z;
    float rotation_offset_x;
    float rotation_offset_z;
    float radius;
    float height;
    float width;
    float depth;
} CollisionShape;

typedef struct { 
    float health;
    float pos_x;
    float pos_y;
    float pos_z;
    float vel_x;
    float vel_y;
    float vel_z;
    float facing;
    int current_move;
    int current_action_frames;
    int animation_frame;
    int frame_advantage;
    int state;
    CollisionShape* hurt_shapes;
    CollisionShape* hit_shapes;
} Character;

typedef struct {
    Log log;                     // Required field
    float* observations; // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int size;
    int x;
    int goal;
    Character* characters;
    int num_characters;
    Client* client;
} Fighter;

void init_collision_shapes(CollisionShape* hurt_shapes, CollisionShape* hit_shapes, float facing) {
    hurt_shapes[0].type = SHAPE_CAPSULE;
    hurt_shapes[0].offset_x = 0.0f;
    hurt_shapes[0].offset_y = 1.4f;  // Center at mid-body (half height)
    hurt_shapes[0].offset_z = 0.0f;
    hurt_shapes[0].height = 0.5f;    // Character height
    hurt_shapes[0].radius = 0.2f;    // Body width

    // Left Arm: Horizontal capsule, extends based on facing angle
    hurt_shapes[1].type = SHAPE_CAPSULE;
    hurt_shapes[1].offset_x = 0.0f;
    hurt_shapes[1].offset_y = 1.6f;  // Shoulder height
    hurt_shapes[1].offset_z = -0.25f;
    hurt_shapes[1].rotation_offset_x = 0.0f;
    hurt_shapes[1].rotation_offset_z = 0.0f;
    hurt_shapes[1].height = 1.0f;    // Arm length (as capsule "height" along axis)
    hurt_shapes[1].radius = 0.075f;   // Arm thickness

    // Right Arm: Similar, opposite offset
    hurt_shapes[2].type = SHAPE_CAPSULE;
    hurt_shapes[2].offset_x = 0.0f;
    hurt_shapes[2].offset_y = 1.6f;
    hurt_shapes[2].offset_z = 0.25f;
    hurt_shapes[2].rotation_offset_x = 0.0f;
    hurt_shapes[2].rotation_offset_z = 0.0f;
    hurt_shapes[2].height = 1.0f;
    hurt_shapes[2].radius = 0.075f;

    // Left Leg: Single vertical capsule for both (or split for precision)
    hurt_shapes[3].type = SHAPE_CAPSULE;
    hurt_shapes[3].offset_x = 0.0f;
    hurt_shapes[3].offset_y = 0.5f;  // From ground to hips
    hurt_shapes[3].offset_z = -0.15f;
    hurt_shapes[3].rotation_offset_x = 0.0f;
    hurt_shapes[3].rotation_offset_z = 0.0f;
    hurt_shapes[3].height = 0.8f;    // Leg height
    hurt_shapes[3].radius = 0.1f;   // Leg width 
    // Right Leg
    hurt_shapes[4].type = SHAPE_CAPSULE;
    hurt_shapes[4].offset_x = 0.0f;
    hurt_shapes[4].offset_y = 0.5f;  // From ground to hips
    hurt_shapes[4].offset_z = 0.15f;
    hurt_shapes[4].rotation_offset_x = 0.0f;
    hurt_shapes[4].rotation_offset_z = 0.0f;
    hurt_shapes[4].height = 0.8f;    // Leg height
    hurt_shapes[4].radius = 0.1f;   // Leg width

    // Head
    hurt_shapes[5].type = SHAPE_CAPSULE;
    hurt_shapes[5].offset_x = 0.0f;
    hurt_shapes[5].offset_y = 2.0f;
    hurt_shapes[5].offset_z = 0.0f;
    hurt_shapes[5].height = 0.2f;    // Head height
    hurt_shapes[5].radius = 0.2f;    // Head width

    // For other shapes (hit_shapes), set to invalid until a move activates them
    for (int i = 0; i < 6; i++) {
        hit_shapes[i].type = SHAPE_NONE;  // Add SHAPE_NONE to enum if needed
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
    character->state = STATE_IDLE;
    character->hurt_shapes = (CollisionShape*)calloc(6, sizeof(CollisionShape));
    character->hit_shapes = (CollisionShape*)calloc(6, sizeof(CollisionShape));
    init_collision_shapes(character->hurt_shapes, character->hit_shapes, character->facing);
}

void init(Fighter* env) {
    env->num_characters = 2;
    env->characters = (Character*)calloc(env->num_characters, sizeof(Character));
    init_character(&env->characters[0], 0);
    init_character(&env->characters[1], PI);
    printf("init\n");
}

void c_reset(Fighter* env) {
    init_collision_shapes(env->characters[0].hurt_shapes, env->characters[0].hit_shapes, 1);
    init_collision_shapes(env->characters[1].hurt_shapes, env->characters[1].hit_shapes, -1);
    printf("reset\n");
}

void sidestep(Fighter* env, Character* character, float direction, int target_index) {
    float sidestep_radius = 4.0f;
    float sidestep_speed = 0.05f;
    float cos_facing = cos(character->facing);
    float sin_facing = sin(character->facing);
    // Calculate current position relative to circle center
    float center_x = character->pos_x + sidestep_radius * cos_facing;  // Circle center X
    float center_z = character->pos_z + sidestep_radius * sin_facing;  // Circle center Z
    
    // Update facing for circular motion
    character->facing += sidestep_speed * direction;
    float new_cos_facing = cos(character->facing);
    float new_sin_facing = sin(character->facing);
    // Calculate new position on circle circumference
    character->pos_x = (center_x) - (sidestep_radius * new_cos_facing);
    character->pos_z = (center_z) - (sidestep_radius * new_sin_facing);
    // new facing toward target
    character->facing = atan2(env->characters[target_index].pos_z - character->pos_z, env->characters[target_index].pos_x - character->pos_x);
    env->characters[target_index].facing = atan2(character->pos_z - env->characters[target_index].pos_z, character->pos_x - env->characters[target_index].pos_x);
}

void move_character(Fighter* env, int character_index, int target_index, int action) {
    Character* character = &env->characters[character_index];
    if (action == ACTION_LEFT) {
        character->pos_x -= 0.15 * cos(character->facing);
        character->pos_z -= 0.15 * sin(character->facing);
    } else if (action == ACTION_RIGHT) {
        character->pos_x += 0.15 * cos(character->facing);
        character->pos_z += 0.15 * sin(character->facing);
    } else if (action == ACTION_UP){
        sidestep(env, character, 1, target_index);
    } else if (action == ACTION_DOWN){
        sidestep(env, character, -1, target_index);
    } 
}

void adjust_skeleton(Fighter* env, int character_index) {
    Character* character = &env->characters[character_index];
    for (int i = 0; i < 6; i++) {   
        if(i == 1 || i == 2 || i == 3 || i == 4){  // Fixed the condition
            CollisionShape* shape = &character->hurt_shapes[i];
            // Store rotated positions in rotation_offset fields, preserve original offsets
            shape->rotation_offset_x = shape->offset_x * cos(character->facing) - shape->offset_z * sin(character->facing);
            shape->rotation_offset_z = shape->offset_x * sin(character->facing) + shape->offset_z * cos(character->facing);
        }
    }
}

void c_step(Fighter* env) {
    for(int i = 0; i < env->num_characters; i++) {
        env->rewards[i] = 0;
        env->terminals[i] = 0;
        move_character(env, i, (i + 1) % env->num_characters, env->actions[i]);
        adjust_skeleton(env, i);
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
    InitWindow(client->width, client->height, "PufferLib Fighter");
    SetTargetFPS(60);
    client->puffers = LoadTexture("resources/puffers_128.png");
    
    client->default_camera_position = (Vector3){ 
        0,           // Same X as target
        10.0f,   // 20 units above target
        20.0f    // 20 units behind target
    };
    client->default_camera_target = (Vector3){0, 0, 0};
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    // DisableCursor(); 
    return client;
}

void c_render(Fighter* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;

    UpdateCamera(&client->camera, CAMERA_FREE);
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);

    // Draw a simple ground plane for reference (XZ plane at Y=0)
    DrawPlane((Vector3){0.0f, 0.0f, 0.0f}, (Vector2){20.0f, 20.0f}, LIGHTGRAY);

    // Draw grid for depth perception
    DrawGrid(20, 1.0f);
    // Render each character's hurt_shapes as capsules (visualize the fighters)
    for (int c = 0; c < env->num_characters; c++) {
        Character* chara = &env->characters[c];
        Color fighter_color = (c == 0) ? RED : BLUE;  // Differentiate players
        
        for (int i = 0; i < 6; i++) {  // Assuming 4 hurt_shapes
            CollisionShape* shape = &chara->hurt_shapes[i];
            if (shape->type != SHAPE_CAPSULE) continue;

            // Compute absolute positions, adjusted by character's pos and facing
            Vector3 center = (Vector3){
                chara->pos_x + shape->offset_x,
                chara->pos_y + shape->offset_y,
                chara->pos_z + shape->offset_z
            };
            DrawSphere(center, shape->radius+0.01,  BLACK);

            // Determine start and end points based on orientation
            // Assume: Torso/Legs vertical (along Y), Arms horizontal (along X)
            Vector3 start_pos, end_pos;
            if (i == 0) {  // Vertical (torso/legs): start at bottom, end at top
                start_pos = (Vector3){ center.x, center.y - (shape->height / 2.0f), center.z };
                end_pos = (Vector3){ center.x, center.y + (shape->height / 2.0f), center.z };
            } else if (i==1 || i==2){
                // Arms: use pre-computed rotation offsets from adjust_skeleton
                start_pos = (Vector3){ 
                    chara->pos_x + shape->rotation_offset_x,
                    center.y, 
                    chara->pos_z + shape->rotation_offset_z
                };
                
                // Arm extends outward from shoulder in the facing direction
                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                end_pos = (Vector3){ 
                    start_pos.x + (shape->height / 2.0f) * facing_x, 
                    center.y, 
                    start_pos.z + (shape->height / 2.0f) * facing_z 
                };
            } else if (i==3 || i==4){
                // Legs: hips rotate with body, feet stay planted
                float stance_offset = 0.1f;
                float facing_x = cos(chara->facing);
                float facing_z = sin(chara->facing);
                if (i == 3) {  // Left leg
                    Vector3 hip_pos = (Vector3){
                        chara->pos_x + shape->rotation_offset_x,
                        center.y + (shape->height / 2.0f),  // Top of leg (hip level)
                        chara->pos_z + shape->rotation_offset_z
                    };
                    
                    Vector3 foot_pos = (Vector3){
                        hip_pos.x + stance_offset * facing_x,  // Original offset, no rotation
                        center.y - (shape->height / 2.0f),  // Bottom of leg (foot level)
                        hip_pos.z + stance_offset * facing_z   // Original offset, no rotation
                    };
                    start_pos = foot_pos;   // Bottom of leg (foot)
                    end_pos = hip_pos;      // Top of leg (hip)
                } else {  // Right leg
                    Vector3 hip_pos = (Vector3){
                        chara->pos_x + shape->rotation_offset_x,
                        center.y + (shape->height / 2.0f),  // Top of leg (hip level)
                        chara->pos_z + shape->rotation_offset_z
                    };
                    Vector3 foot_pos = (Vector3){
                        hip_pos.x - stance_offset * facing_x,  // Original offset, no rotation
                        center.y - (shape->height / 2.0f),  // Bottom of leg (foot level)
                        hip_pos.z - stance_offset * facing_z   // Original offset, no rotation
                    };
                    start_pos = foot_pos;   // Bottom of leg (foot)
                    end_pos = hip_pos;      // Top of leg (hip)
                }
            } else {
                // Head: stays centered
                start_pos = (Vector3){ center.x, center.y, center.z };
                end_pos = (Vector3){ center.x, center.y, center.z };
            }
            if(i ==1) fighter_color = PINK;
            if(i ==2) fighter_color = GREEN;
            if(i ==3) fighter_color = YELLOW;
            if(i ==4) fighter_color = PURPLE;
            // Draw the capsule (solid for body, or use DrawCapsuleWires for outline)
            DrawCapsule(start_pos, end_pos, shape->radius, 16, 8, fighter_color);  // 16 slices, 8 rings for smoothness
            DrawCapsuleWires(start_pos, end_pos, shape->radius, 16, 8, BLACK);  // Wireframe for debug
        }
    }
    EndMode3D();
    
    // Draw health bars on top of screen (2D overlay)
    float health_bar_width = 200.0f;
    float health_bar_height = 20.0f;
    float health_bar_y = 20.0f;
    float health_bar_margin = 20.0f;
    
    // Character 1 health bar (top left)
    float char1_x = health_bar_margin;
    float char1_health_ratio = env->characters[0].health / 100.0f;
    Rectangle char1_background = {char1_x, health_bar_y, health_bar_width, health_bar_height};
    Rectangle char1_health = {char1_x, health_bar_y, health_bar_width * char1_health_ratio, health_bar_height};
    DrawRectangleRec(char1_background, DARKGRAY);
    DrawRectangleRec(char1_health, RED);
    DrawRectangleLinesEx(char1_background, 2, WHITE);
    DrawText(TextFormat("Player 1: %d", (int)env->characters[0].health), char1_x + 10, health_bar_y + 25, 20, WHITE);
    
    // Character 2 health bar (top right)
    float char2_x = client->width - health_bar_width - health_bar_margin;
    float char2_health_ratio = env->characters[1].health / 100.0f;
    Rectangle char2_background = {char2_x, health_bar_y, health_bar_width, health_bar_height};
    Rectangle char2_health = {char2_x, health_bar_y, health_bar_width * char2_health_ratio, health_bar_height};
    DrawRectangleRec(char2_background, DARKGRAY);
    DrawRectangleRec(char2_health, BLUE);
    DrawRectangleLinesEx(char2_background, 2, WHITE);
    DrawText(TextFormat("Player 2: %d", (int)env->characters[1].health), char2_x + 10, health_bar_y + 25, 20, WHITE);
    EndDrawing();
}

void c_close(Fighter* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
