#pragma once

#include <iostream>
#include <ctime>
#include <cstdlib>

#include "cshogi.h"
#include "types.hpp"

// TODO
//     : ���ۃN���X�Ƃ������A���N���X�����B
//       �ق�ł����āAoverride �I�Ȃ��Ƃ��āA����Player�Ƃ��ăI�u�W�F�N�g�͎����Ă邯�ǁA
//       random_player �̂��Ăяo���A�I�Ȃ̂��������B
//     : �Ȃ񂩁Aheader �̒��g�ς����������ƁA.exe �ɔ��f�����B
//     : �{���́A�w�����Ԃ�_go(), readyok���ۂ���Ԃ�_isready() ��gensfen �����邤���ŕK�v�Ȃ񂾂��ǁA
//       ����͕��񉻂Ƃ��Aponder�Ƃ��������Ή�������ɔ���B

class BasePlayer {
protected:
public:
	BasePlayer() {
	}

	virtual ~BasePlayer() {}

	virtual void usi() = 0;
	virtual void setoption(std::string cmd) = 0;
	virtual void isready() = 0;                                 // �Ō��readyok ��\��
	virtual void usinewgame() = 0;
	virtual void position(const std::string& moves_str) = 0;    // position_str ��set
	virtual void go() = 0;                                      // �Ō��bestmove ��\��
};

