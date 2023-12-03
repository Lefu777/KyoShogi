#include "gensfen_parser.hpp"


// gensfenrand -n 10000000000 -save_interval 10000000 -draw 256 -file data/sfen
#ifdef GEN_SFEN
void gensfen_loop() {
	while (true) {
		std::string cmd, token;
		getline(std::cin, cmd);
		std::istringstream iss(cmd);
		iss >> token;

		if (token == "mode") {
			std::cout << "mode = GEN_SFEN" << std::endl;
		}
		else if (token == "gensfenrand") {
			bool is_ok = true;    // �������J�n���Ă悢���ۂ��̃t���O
			unsigned long long n_gensfens = -1;
			unsigned long long save_interval = -1;
			int max_moves_to_draw = -1;
			std::string file_name = "teacher";

			// TODO: thread �g��
			while (iss >> token) {    // ��ǂݏo�������A�ǂ�
				if (token == "-n") {
					iss >> token;
					n_gensfens = std::stoull(token);
				}
				else if (token == "-save_interval") {
					// TODO: ����A����iss ����ULL �ɓ����Ȃ����ˁB
					iss >> token;
					save_interval = std::stoull(token);
				}
				else if (token == "-draw") {
					iss >> token;
					max_moves_to_draw = std::stoi(token);
				}
				else if (token == "-file") {
					iss >> token;
					file_name = token;
				}
				else if (token == "--help") {
					std::cout << "ex1>" << std::endl;
					std::cout << "* 100���ǖʐ����A1000�����ɕۑ��A256���������" << std::endl;
					std::cout << "* ���̎��Adata �t�H���_�͂��炩���ߍ쐬���Ă�������" << std::endl;
					std::cout << "gensfenrand -n 10000000000 -save_interval 10000000 -draw 256 -file data/sfen" << std::endl;
					std::cout << "ex2>" << std::endl;
					std::cout << "gensfenrand --help" << std::endl;
					is_ok = false;
				}
				else {
					std::cout << "ParserError: parser got unexpected token == [" << token << "]" << std::endl;;
					exit(1);
				}
			}
			if (is_ok) {
				GensfenRandom generator(n_gensfens, save_interval, max_moves_to_draw, file_name);
				generator.gen();
			}
		}
		//else if (token == "gensfen") {
		//	;
		//}
		else if (token == "quit") {
			exit(1);
		}
		else {
			std::cout << "ParserError: parser got unexpected token == [" << token << "]" << std::endl;;
			exit(1);
		}
	}
}
#endif