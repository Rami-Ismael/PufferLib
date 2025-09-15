#include "fighter.h"

int main() {
    Fighter env = {.num_characters = 2};    
    env.observations = (float*)calloc(1, sizeof(float));
    env.actions = (int*)calloc(env.num_characters, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(&env);
    c_reset(&env);
    c_render(&env);
    printf("render done\n");
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_LEFT)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_RIGHT)) {
                env.actions[0] = 2;
            } else if (IsKeyDown(KEY_UP)) {
                env.actions[0] = 3;
            } else if (IsKeyDown(KEY_DOWN)) {
                env.actions[0] = 4;
            } else if (IsKeyPressed(KEY_SPACE)) {
                env.actions[0] = 5;
            }
            else {
                env.actions[0] = 0;
            }
        }
        else {
            env.actions[0] = 0;
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

