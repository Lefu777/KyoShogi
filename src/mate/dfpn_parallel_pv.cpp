#include "dfpn_parallel_pv.hpp"
#include "MT.hpp"


template<bool root_is_self>
void ParallelPvDfpnThread::_run(__Board& board, UctNode* node) {
	if (this->is_stop()) {
		return;
	}

	// HACK: is_evaled() ならis_expanded() されてないとおかしいはずで、条件は前者だけで良いはず。。。
	// dlshogi、ここで!node->child とかしてるけど、
	// expand_node のmake_unique と被ったら普通にデータ競合だと思うんよね。
	// 未評価だとucb スコアを計算できないから最善の子ノード計算できないし、
	// 未展開だとそもそも子ノードが無いので、ここで終了。
	if (!node->is_evaled() || !node->is_expanded()) {
		std::this_thread::yield();
		return;
	}

	// select_max_ucb_child() は、expand_node(), eval_node() より後に実行されるならスレッドセーフ。
	// データ競合する可能性のあるやつにはアクセスせんはずで、参照しても良いはず。。。
#if 1
	const auto& next_index = select_max_ucb_child(nullptr, node);
#else
	// debug
	volatile const auto next_index = select_max_ucb_child(nullptr, node);
#endif

	if (next_index < 0) {
		std::this_thread::yield();
		return;
	}

	auto& next_child_info = node->child_infos[next_index];
	// 詰みなら、ここで終了。
	if (next_child_info.is_win() || next_child_info.is_lose()) {
		std::this_thread::yield();
		return;
	}

	//volatile const int lm_size = node->legal_moves_size;
	//volatile const std::string sfen_parent = board.toSFEN();
	//volatile const Move next_child_info_move = next_child_info.move;
	//volatile const std::string next_child_info_move_usi = next_child_info.move.toUSI();

	board.push(next_child_info.move);

	//volatile const std::string sfen_child = board.toSFEN();


	if (this->is_stop()) {
		return;
	}

	_shared_searched_nodes_mtx->lock();
	const auto& status = _shared_searched_nodes->emplace(&next_child_info);
	_shared_searched_nodes_mtx->unlock();


	if (status.second) {    // 未探索
		if constexpr (root_is_self) {    // 子供は敵側
			_dfpn_group.set_tt(_enem_TT);
		}
		else {
			_dfpn_group.set_tt(_self_TT);
		}

		_dfpn_group.set_root_board(board);
		_dfpn_group.run();
		_dfpn_group.join();
		const bool stop = this->is_stop();
		if (stop && _dfpn_group.get_pn() != 0 && _dfpn_group.get_dn() != 0) {
			// "終了直後" でstop のフラグが立っていて、且つpn, dn が中途半端な値の場合、
			// (ノード数制限ではなく)強制終了で中断された可能性が高いので未探索のままとする。
			_shared_searched_nodes_mtx->lock();
			// FIX: 何故これがだめ。
			const auto tmp = &next_child_info;
			_shared_searched_nodes->erase(tmp);
			_shared_searched_nodes_mtx->unlock();
		}
		else {
			if (_dfpn_group.get_is_mate()) {
				//sync_cout << "info strig PvDfpn: win, ply=" << board.ply() << ",key=" << board.getKey() << sync_endl;
				next_child_info.set_win();
			}
		}
	}
	else {    // 探索済みなら次へ
		if (next_child_info.is_exist()) {    // Node が存在する場合のみ遷移可能
			_run<!root_is_self>(board, node->child_nodes[next_index].get());
		}
		else {
			std::this_thread::yield();
		}
	}

	// TODO: 本当はここいらでpop() して、board を再利用すべき。
}

void ParallelPvDfpnThread::run() {
	_stop.store(false);

	//int randnum = mt_genrand_int32() % 10000;
	//std::this_thread::sleep_for(std::chrono::microseconds(10000));

	_th = std::thread(
		[&]() {
			while (!this->is_stop()) {
				//sync_cout << "info string ParallelPvDfpnThread start: " << randnum << sync_endl;
				__Board board_cpy(_root_board);
				this->_run<true>(board_cpy, get_global_tree()->current_head);
				//sync_cout << "info string ParallelPvDfpnThread done : " << randnum << sync_endl;
			}
		}
	);
}

template void ParallelPvDfpnThread::_run<true>(__Board& board, UctNode* node);
template void ParallelPvDfpnThread::_run<false>(__Board& board, UctNode* node);
