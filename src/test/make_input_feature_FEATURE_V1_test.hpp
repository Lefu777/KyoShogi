#pragma once
#include "config.hpp"
#include "cshogi.h"

#ifdef FEATURE_V1
// 盤上の駒について、フラグを立てる。
// 色がcのplyaer の駒pt がsq に存在するというフラグを立てる。
inline void set_feature1_test(float* dst, const Color c, const PieceType pt, const Square sq) {
    // NOTE
    //     : PieceType は、occupied == 0, Pawn == 1, Lnace == 2, ...となっている。
    //       今回はoccupied は使わないので、その分を引いて(pt -1)としている。
    //const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (u32)(pt - 1) * N_FEATURE_WH + (u32)sq;
    //const u32 idx =
    //	0                                      // feature1 より前の特徴量の総数
    //	+ (u32)c * N_FEATURE1_WHC_PER_COLOR
    //	+ (u32)(pt - 1) * N_FEATURE_WH
    //	+ (u32)sq;
    const u32&& idx = (
        0 +
        (u32)c * N_FEATURE1_CHANNEL_PER_COLOR +
        (u32)(pt - 1)
    ) * N_FEATURE_WH + sq;
    dst[idx] += 1;
}

// 持ち駒について、フラグを立てる。
// num枚のチャンネル全てにフラグを立てる。(81 * num個のフラグを立てる。)
// @arg c: 与えられた持ち駒hp を持っている方のplayerの色
// @arg hp: 持ち駒の種類
// @arg num: 持ち駒の数
inline void set_feature2_test(float* dst, const Color c, const HandPiece hp, const u32 num) {
    // 引数で渡されたhp の手前までの、各駒の持ち駒の上限数の和。
    // つまり、hp の持ち駒についてのチャンネルの先頭のchannel index.(チャンネル番号)
    const u32&& hp_channel = std::accumulate(std::begin(N_MAX_HANDS), std::next(std::begin(N_MAX_HANDS), (u32)hp), 0);
    //const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (N_FEATURE1_CHANNEL_PER_COLOR + tmp) * N_FEATURE_WH;
    //const u32 idx =
    //	N_FEATURE1_WHC         // feature1 より前の特徴量の総数
    //	+ (u32)c * N_FEATURE2_WHC_PER_COLOR
    //	+ hp_channel_start_idx * N_FEATURE_WH;
    const u32&& idx = (
        N_FEATURE1_CHANNEL +
        (u32)c * N_FEATURE2_CHANNEL_PER_COLOR +
        hp_channel
    ) * N_FEATURE_WH;

    for (int i = 0; i < N_FEATURE_WH * num; ++i) {
        dst[idx + i] += 1;
    }
}

// 入力特徴量全体が、先に0初期化されているならば、in_check = true の時だけstore すれば良い。
template<bool in_check>
inline void set_feature3_test(float* dst) {
    if constexpr (in_check) {
        constexpr u32 idx = (
            N_FEATURE1_CHANNEL + N_FEATURE2_CHANNEL
        ) * N_FEATURE_WH;
        for (int i = 0; i < N_FEATURE_WH; ++i) {
            dst[idx + i] += 1;
        }
    }
    else {
        throw std::runtime_error();
    }
}

// 成功(@20230925)
// 成功することは、バグっていないことの必要条件
inline void make_input_features_FATURE_V1_test() {
    float* dst = new float[N_FEATURE_WHC];

    std::fill_n(dst, N_FEATURE_WHC, 0);    // ひとまず全部0 埋めする。後から必要な箇所だけフラグを立てる。

    //////// 以下の処理によって、dst の全要素が1 になる。それ超過でもそれ未満でもなく。
    //// 盤上の駒
    for (Color c = Black; c < ColorNum; ++c) {
        for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                set_feature1_test(dst, c, pt, sq);
            }
        }
    }

    //// 持ち駒
    for (Color c = Black; c < ColorNum; ++c) {
        for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
            int num = N_MAX_HANDS[hp];    // 上限値
            set_feature2_test(dst, c, hp, num);
        }
    }

    ////// 王手
    set_feature3_test<true>(dst);

    //// check
    std::cout << "info: N_FEATURE1_CHANNEL = " << N_FEATURE1_CHANNEL << std::endl;
    std::cout << "info: N_FEATURE2_CHANNEL = " << N_FEATURE2_CHANNEL << std::endl;
    std::cout << "info: N_FEATURE_WHC = " << N_FEATURE_WHC << std::endl;
    bool failed = false;
    for (int i = 0; i < (N_FEATURE1_CHANNEL + N_FEATURE2_CHANNEL + N_FEATURE3_CHANNEL) * N_FEATURE_WH; ++i) {
        if (dst[i] != 1) {
            failed |= true;
            std::cout << "Error: dst[" << i << "] = " << dst[i] << std::endl;
        }
    }

    delete[] dst;

    if (!failed) {
        std::cout << "info: test is successful." << std::endl;
    }
    else {
        std::cout << "info: test failed." << std::endl;
    }
}
#endif // FEATURE_V1