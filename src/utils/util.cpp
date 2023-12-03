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
	// std::stringにして値を返す
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
	// std::stringにして値を返す
	return ss.str();
}

void write_teacher(
	const std::vector<teacher_t>& teachers,
	std::string base_file_name,
	int output_interval,    // 書き出し中の表示を更新する間隔(のはずだったが、現状使ってない。)
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
		//    : 文字列の書き込みテスト。(fwrite とか使った方がラクな気がしてきた。cmnist で一回やってるしさ。)
		//    ; read_file() が必要。ほんでもって、read_file は関数で、write_file は関数じゃないのはなんかきもい。
		//      file_operator.hpp とかに纏めることは出来ないだろうか。
		//    : read_file() に関しては、cython つかってpython でnumpy に最終的に格納したい。
		//      まぁ最悪別の関数として作っても良いけどね。c++用、python用 と別々で。
		//   -> というか。bitboard 使わないと特徴量作成出来ない気がしてまして。
		//      仮にposition.set_sfen() とsfen 形式でset 可能なら、一旦それをfor文で回す感じにしようかしら？
		//      可能なら、dlshogi のmake_feature を見たい。bitboard 使ってんのか、はたまた別なのか。
		//    ; それらが出来たらpytorch でモデル定義して学習。
		//      モデル定義どうするか悩ましいね。入・出力特徴量とか含めて色々。
		//    : 恐らくsfen の文字数を先に書き込んでおかないと、読み込みが出来んで。
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

// HACK: 未使用だよね....?
// 自作dtype型の配列に突っ込む方法。cshogi のファイル検索したら出てくるやろ。
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

