#include "util.hpp"

float my_round(float x, int d) {
	return std::round(x * pow(10, d)) / pow(10, d);
}

std::string get_now_str_for_filename() {
	time_t t = time(nullptr);
	tm localTime;
	if (localtime_s(&localTime, &t) != 0) {
		std::cout << "Error: failed to localtime_s()" << std::endl;
	}
	std::ostringstream ss;
	ss << localTime.tm_year + 1900;
	ss << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
	ss << std::setw(2) << std::setfill('0') << localTime.tm_mday;
	ss << std::setw(2) << std::setfill('0') << localTime.tm_hour;
	ss << std::setw(2) << std::setfill('0') << localTime.tm_min;
	ss << std::setw(2) << std::setfill('0') << localTime.tm_sec;
	// std::string�ɂ��Ēl��Ԃ�
	return ss.str();
}

std::string get_now_str() {
	time_t t = time(nullptr);
	tm localTime;
	if (localtime_s(&localTime, &t) != 0) {
		std::cout << "Error: failed to localtime_s()" << std::endl;
	}
	std::ostringstream ss;
	ss << localTime.tm_year + 1900
		<< "/" << localTime.tm_mon + 1
		<< "/" << localTime.tm_mday
		<< " " << localTime.tm_hour
		<< ":" << localTime.tm_min
		<< ":" << localTime.tm_sec;
	// std::string�ɂ��Ēl��Ԃ�
	return ss.str();
}

void write_teacher(
	const std::vector<teacher_t>& teachers,
	std::string base_file_name,
	int output_interval,    // �����o�����̕\�����X�V����Ԋu(�̂͂����������A����g���ĂȂ��B)
	bool random_name,
	bool verbose
) {
	std::string str_random_num = std::to_string(mt_genrand_int32());
	std::stringstream ss;
	ss << base_file_name;
	if (random_name) {
		ss << "_" << get_now_str_for_filename()
		    << "_" << std::hex << str_random_num;
	}
	ss << ".bin";

	FILE* file = nullptr;
	if ((fopen_s(&file, ss.str().c_str(), "wb")) != 0) {
		std::cout << "Error: [write_teacher()] failed to open." << ss.str() << std::endl;
		exit(1);
	}

	for (const auto& t : teachers) {
		// TODO
		//    : ������̏������݃e�X�g�B(fwrite �Ƃ��g�����������N�ȋC�����Ă����Bcmnist �ň�����Ă邵���B)
		//    ; read_file() ���K�v�B�ق�ł����āAread_file �͊֐��ŁAwrite_file �͊֐�����Ȃ��̂͂Ȃ񂩂������B
		//      file_operator.hpp �Ƃ��ɓZ�߂邱�Ƃ͏o���Ȃ����낤���B
		//    : read_file() �Ɋւ��ẮAcython ������python ��numpy �ɍŏI�I�Ɋi�[�������B
		//      �܂��ň��ʂ̊֐��Ƃ��č���Ă��ǂ����ǂˁBc++�p�Apython�p �ƕʁX�ŁB
		//   -> �Ƃ������Bbitboard �g��Ȃ��Ɠ����ʍ쐬�o���Ȃ��C�����Ă܂��āB
		//      ����position.set_sfen() ��sfen �`����set �\�Ȃ�A��U�����for���ŉ񂷊����ɂ��悤������H
		//      �\�Ȃ�Adlshogi ��make_feature ���������Bbitboard �g���Ă�̂��A�͂��܂��ʂȂ̂��B
		//    ; ����炪�o������pytorch �Ń��f����`���Ċw�K�B
		//      ���f����`�ǂ����邩�Y�܂����ˁB���E�o�͓����ʂƂ��܂߂ĐF�X�B
		//    : ���炭sfen �̕��������ɏ�������ł����Ȃ��ƁA�ǂݍ��݂��o����ŁB
		auto sfen_size = t.sfen.size();
		fwrite(&sfen_size, sizeof(size_t), 1, file);
		fwrite(t.sfen.c_str(), sizeof(char), sfen_size, file);
		fwrite(&t.move, sizeof(int), 1, file);
		fwrite(&t.ply, sizeof(int), 1, file);
		fwrite(&t.value, sizeof(float), 1, file);
		fwrite(&t.result, sizeof(float), 1, file);
	}

	if (verbose) {
		std::cout << get_now_str() << " : write done." << std::endl;
	}
	fclose(file);
}

// HACK: ���g�p�����....?
// ����dtype�^�̔z��ɓ˂����ޕ��@�Bcshogi �̃t�@�C������������o�Ă�����B
void read_teacher(char* ndarray, size_t buf_size, std::string file_path) {
	std::vector<teacher_t> teachers;
	FILE* file;
	if ((fopen_s(&file, file_path.c_str(), "rb")) != 0) {
		// https://qiita.com/izuki_y/items/26bf20c4b3b3750ab7a7
		std::cout << "Error: [read_teacher()] failed to open."  << std::endl;
		exit(1);
	}



}

std::vector<std::string> tokenize(std::string cmd) {
	std::istringstream iss(cmd);
	std::string token;
	std::vector<std::string> tokens;
	while (iss >> token) {
		tokens.emplace_back(token);
	}
	return tokens;
}

// --------------------
//  sync_out/sync_endl
// --------------------

std::ostream& operator<<(std::ostream& os, YoSyncCout sc) {

	static std::mutex m;

	if (sc == IO_LOCK)
		m.lock();

	if (sc == IO_UNLOCK)
		m.unlock();

	return os;
}

