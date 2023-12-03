#pragma once

#ifndef USI_PARSER_H
#define USI_PARSER_H

#include <iostream>
#include <sstream>
#include <string>

#include "config.hpp"


#ifdef USI_ENGINE
#include "base_player.hpp"

#ifdef RANDOM_PLAYER
#include "random_player.hpp"
#elif defined UCT_RANDOM_PLAYER
#include "uct_random_player.hpp"
#elif defined PUCT_PLAYER
#include "puct_player.hpp"
#elif defined PARALLEL_PUCT_PLAYER
#include "parallel_puct_player.hpp"
#endif

void usi_loop();
#endif

#endif