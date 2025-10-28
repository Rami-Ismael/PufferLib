#include "fighter.h"

#define Env Fighter
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = unpack(kwargs, "num_agents");
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->selfplay = unpack(kwargs, "selfplay");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
