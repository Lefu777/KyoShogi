#include "uct_node.hpp"

#if defined(RANDOM_PLAYER) || defined(UCT_RANDOM_PLAYER) || defined(PUCT_PLAYER)

// https://learn.microsoft.com/ja-jp/cpp/error-messages/tool-errors/linker-tools-error-lnk2005?view=msvc-170
std::unique_ptr<UctTree> global_tree;

#endif