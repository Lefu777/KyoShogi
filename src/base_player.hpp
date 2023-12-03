#pragma once

#include <iostream>
#include <ctime>
#include <cstdlib>

#include "cshogi.h"
#include "types.hpp"

// TODO
//     : 抽象クラスというか、基底クラスを作る。
//       ほんでもって、override 的なことして、基底のPlayerとしてオブジェクトは持ってるけど、
//       random_player のを呼び出す、的なのをしたい。
//     : なんか、header の中身変えただけだと、.exe に反映されん。
//     : 本当は、指し手を返す_go(), readyokか否かを返す_isready() もgensfen をするうえで必要なんだけど、
//       これは並列化とか、ponderとかもろもろ対応した後に縛る。

class BasePlayer {
protected:
public:
	BasePlayer() {
	}

	virtual ~BasePlayer() {}

	virtual void usi() = 0;
	virtual void setoption(std::string cmd) = 0;
	virtual void isready() = 0;                                 // 最後はreadyok を表示
	virtual void usinewgame() = 0;
	virtual void position(const std::string& moves_str) = 0;    // position_str にset
	virtual void go() = 0;                                      // 最後はbestmove を表示
};

