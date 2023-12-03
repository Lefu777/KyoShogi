#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef NOMINMAX
#include <filesystem>
#include "cshogi.h"
#include "types.hpp"

// ����dtype�^�̔z��ɓ˂����ޕ��@�Bcshogi �̃t�@�C������������o�Ă�����B
int read_teacher_test0(std::string file_path);

// https://learn.microsoft.com/ja-jp/windows/win32/fileio/listing-the-files-in-a-directory
// @arg dir: ex> dir = "./folder_name"
int read_teacher_test1(std::string dir);

// HACK: ��΂���A��������Ȃ���ȁB�B�B
// https://qiita.com/brackss1/items/e92da6458172397f7225
// �����o�ϐ���int member; ��݂̂̂���N���XTestClass() ���������Ƃ��悤�B
// std::vector<TestClass> vec;
// TestClass tmp(5);
// ���̎��A�ȉ��̓�̏ꍇ���l������B
// vec.push_back(tmp);
// vec.push_vack(12)
// ���x�͂��ꂼ��ȉ��ƂȂ����B
// 23968
// 24043
// 23901
inline int vector_back_test(unsigned long long n_loops) {
	std::vector<int> vs1;
	std::vector<int> vs2;
	std::vector<int> vs3;
	int tmp = 5;

	LARGE_INTEGER freq, start, end;
	double duration_ms;

#if false
	if (!QueryPerformanceFrequency(&freq)) {    // �P�ʏK��
		return 0;
	}
	if (!QueryPerformanceCounter(&start)) {
		return 0;
	}
	for (unsigned long long i = 0; i < n_loops; ++i) {
		vs1.emplace_back(5);
	}
	if (!QueryPerformanceCounter(&end)) {
		return 0;
	}
	duration_ms = 1000 * ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart);
	std::cout << "duration(emplace_back(rv)) = " << duration_ms << " [ms]\n";
#endif
#if 1
	if (!QueryPerformanceFrequency(&freq)) {    // �P�ʏK��
		return 0;
	}
	if (!QueryPerformanceCounter(&start)) {
		return 0;
	}
	for (unsigned long long i = 0; i < n_loops; ++i) {
		vs2.emplace_back(tmp);
	}
	if (!QueryPerformanceCounter(&end)) {
		return 0;
	}
	duration_ms = 1000 * ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart);
	std::cout << "duration(emplace_back(lv)) = " << duration_ms << " [ms]\n";

	std::cout << "vs2[0] = " << vs2[0] << std::endl;    // 5
	tmp = 9;
	std::cout << "vs2[0] = " << vs2[0] << std::endl;    // 5

	std::string x(3, 'x');
	std::cout << x << std::endl;

	char buf[100];
	buf[0] = 'a';
	buf[1] = 'b';
	buf[2] = 'c';
	buf[3] = '\0';
	std::string str1(buf, 10);
	std::string str2(buf, 4);
	std::string str3(buf, 3);
	std::cout << buf << std::endl;
	std::cout << "[" << str1 << "]" << std::endl;    // ���炩�ɂ���Ă̓_���B
	std::cout << "[" << str2 << "]" << std::endl;    // [abc]
	std::cout << "[" << str3 << "]" << std::endl;    // [abc]

#endif
#if false
	if (!QueryPerformanceFrequency(&freq)) {    // �P�ʏK��
		return 0;
	}
	if (!QueryPerformanceCounter(&start)) {
		return 0;
	}
	for (unsigned long long i = 0; i < n_loops; ++i) {
		vs2.push_back(tmp);
	}
	if (!QueryPerformanceCounter(&end)) {
		return 0;
	}
	duration_ms = 1000 * ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart);
	std::cout << "duration(push_back(lv)) = " << duration_ms << " [ms]\n";
#endif

	return 1;
}