#include "fighter.h"
#include <time.h>
#include <unistd.h>
int demo() {
    Fighter env = {.num_characters = 2, .num_agents = 1};    
    env.observations = (float*)calloc(env.num_agents * 14, sizeof(float));
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    init(&env);
    //c_reset(&env);
    c_render(&env);
    printf("render done\n");
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[0] = 2;
            } else if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
                env.actions[0] = 3;
            } else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
                env.actions[0] = 4;
            } else if (IsKeyPressed(KEY_SPACE)) {
                env.actions[0] = 5;
            }
            else if (IsKeyPressed(KEY_U)){
                env.actions[0] = 6;
            }
            else if (IsKeyPressed(KEY_I)){
                env.actions[0] = 9;
            }
            else if (IsKeyPressed(KEY_J)){
                env.actions[0] = 7;
            }
            else if (IsKeyPressed(KEY_K)){
                env.actions[0] = 8;
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
    return 0;
}


void test_performance(int timeout) {
    Fighter env = {.num_characters = 2, .num_agents = 1};    
    env.observations = (float*)calloc(env.num_agents*14, sizeof(float));
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    init(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 9;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps*env.num_agents / (end - start);
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

