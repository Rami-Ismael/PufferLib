#include "fighter.h"
#include <time.h>
#include <unistd.h>
int demo() {
    Fighter env = {.num_characters = 2};    
    env.observations = (float*)calloc(env.num_characters, sizeof(float));
    env.actions = (int*)calloc(env.num_characters, sizeof(int));
    env.rewards = (float*)calloc(env.num_characters, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_characters, sizeof(unsigned char));
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


void test_performance(int timeout) {
    Fighter env = {.num_characters = 2};    
    env.observations = (float*)calloc(env.num_characters, sizeof(float));
    env.actions = (int*)calloc(env.num_characters, sizeof(int));
    env.rewards = (float*)calloc(env.num_characters, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_characters, sizeof(unsigned char));
    init(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 3;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps*env.num_characters / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}

int main() {
    demo();
    //test_performance(10);
}

