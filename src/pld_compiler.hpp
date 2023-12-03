#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <numeric>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <queue>
#include <unordered_map>

#include <string.h>


namespace Compiler {
    class Error {
        //std::string aboutLocationName_;    // "lexer" とか"parser" とか。どの部門でのエラーか。だいたいの場所。
        std::queue<std::string> errorQueue_;
        int errorNum_;
        //bool isWarnDirectRecursion_;
        bool isFlush_;
        bool isOk_;

    public:
        void init() {
            isFlush_ = true;
            isOk_ = true;

            std::queue<std::string>().swap(errorQueue_);
        }

        Error()
            :/*aboutLocationName_(""), */errorNum_(0)/*, isWarnDirectRecursion_(true), isFlush_(true), isOk_(true)*/
        {
            init();
        }

        Error(std::string aboutLocationName)
            :/*aboutLocationName_(aboutLocationName), */errorNum_(0)/*,isWarnDirectRecursion_(true), isFlush_(true), isOk_(true)*/
        {
            init();
        }

        // Error が一つも無ければtrue を返す
        bool isOk() {
            // TODO: ERROR あればflush();
            return isOk_;
        }

        void pushMessage(std::string msg) {
            if (isOk_) {
                isOk_ = false;
            }

            errorQueue_.push(msg);
            ++errorNum_;
        }

        //void errorFatal(std::string msg) {
        //	this->pushMessage("FatalERROR: " + msg);
        //	this->flush();
        //	std::cout << "abort compilation." << std::endl;
        //	exit(1);
        //}

        //void flushOnce() {
        //    if (isFlush_) {
        //        this->flush();
        //        isFlush_ = false;
        //    }
        //}

        void flush() {
            flush_with_header("");
        }

        // @arg header
        //     : 各行の先頭に挿入する文字。
        //       ex> "info string "
        void flush_with_header(const std::string header) {
            if (errorNum_ == 0) {
                // std::cout << aboutLocationName_ << " has no error :)" << std::endl;
            }
            else {
                std::cout << "flush " << errorNum_ << " error :-C" << std::endl;

                while (errorNum_--) {
                    std::cout << header << errorQueue_.front() << std::endl;
                    errorQueue_.pop();
                }
            }
        }
    };

    // const
    constexpr char EXT_CHAR = 0x03;               /*　終端を表す　*/
    constexpr int MAXNAME = 127;                  /*　定数,変数,関数名の最大の長さ　*/

    // global 変数
    Error g_error;

    enum class tokenTypeIds {                  /*　キーや文字の種類（名前）　*/
        // 予約語
        Begin, End,
        If, Then,
        While, Do,
        Return, Func,
        Var, Const, Odd,
        Write, WriteLn,
        end_of_KeyWd,                   /*　予約語の名前はここまで　*/

        // 演算子と区切り記号の名前
        Plus, Minus,
        Mult, Div,
        Lparen, Rparen,
        Comma, Period, Semicolon,
        Equal, Lss, Grtr,                /*　1byteに収まる物はここまで　*/
        NotEq, LssEq, GrtrEq,
        Assign,
        end_of_KeySym,                  /*　演算子と区切り記号の名前はここまで　*/

        Ident, Num, Illegal, Ext,              /*　トークンの種類　*/
        end_of_Token,

        letter, digit, colon, others    /*　上記以外の文字の種類　*/
    };
    typedef tokenTypeIds TokenTypeId;
    typedef tokenTypeIds ttid;
    TokenTypeId cppCharTypeId[256];    /*　c++ のchar型(1byte) の値に対応するTokenTypeId の表　*/

    union valueUnion {
        char name[MAXNAME];    /*　Identfierの時、その名前　*/
        int64_t value = 0;             /*　Numの時、その値　*/
    };

    struct token {
        TokenTypeId typeId;
        // https://rainbow-engine.com/cpp_unions_howto/
        valueUnion u;
        // int headPos;               /* 場所を記憶 */
    };
    typedef token Token;

    // NOTE: https://learn.microsoft.com/ja-jp/cpp/c-language/type-char?view=msvc-170
    //       char とASCII は必ずしも対応しているとは限らんみたいな話を聞いたことがあるけど、
    //      これ見る限り対応してそう。
    //     : http://www3.nit.ac.jp/~tamura/ex2/ascii.html
    // 事前に呼ぶ。_next_token() までに呼ばなければならない
    void initcppCharTypeId()    /*　文字の種類を示す表を作る関数　*/
    {
        int i;

        for (i = 0; i < 256; ++i) {  // ひとまず全部others に
            cppCharTypeId[i] = ttid::others;
        }
        for (i = '0'; i <= '9'; ++i) {
            cppCharTypeId[i] = ttid::digit;
        }
        for (i = 'A'; i <= 'Z'; ++i) {
            cppCharTypeId[i] = ttid::letter;
        }
        for (i = 'a'; i <= 'z'; ++i) {
            cppCharTypeId[i] = ttid::letter;
        }
        cppCharTypeId['_'] = ttid::letter;

        cppCharTypeId['+'] = ttid::Plus; cppCharTypeId['-'] = ttid::Minus;
        cppCharTypeId['*'] = ttid::Mult; cppCharTypeId['/'] = ttid::Div;
        cppCharTypeId['('] = ttid::Lparen; cppCharTypeId[')'] = ttid::Rparen;

        cppCharTypeId[','] = ttid::Comma; cppCharTypeId['.'] = ttid::Period;
        cppCharTypeId[';'] = ttid::Semicolon;

        cppCharTypeId['='] = ttid::Equal; cppCharTypeId['<'] = ttid::Lss;
        cppCharTypeId['>'] = ttid::Grtr;

        cppCharTypeId[EXT_CHAR] = ttid::Ext;

        cppCharTypeId[':'] = ttid::colon;
    }

    // 予約語
    // HACK: mapのkeyを、char[MAXNAME] で受け取るとなんか引数の型が一致しないとおっしゃる。
    //       一旦 std::string にしてエラー消した。
    std::unordered_map<std::string, TokenTypeId> keywords{
        //{"begin", ttid::Begin},
        //{"end", ttid::End},
        //{"if", ttid::If},
        //{"then", ttid::Then},
        //{"while", ttid::While},
        //{"do", ttid::Do},
        //{"return", ttid::Return},
        //{"function", ttid::Func},
        //{"var", ttid::Var},
        //{"const", ttid::Const},
        //{"odd", ttid::Odd},
        //{"write", ttid::Write},
        //{"writeln",ttid::WriteLn},
        //{"$dummy1",ttid::end_of_KeyWd},

        ////　記号と名前(KeyId)の表　
        //{"+", ttid::Plus},
        {"-", ttid::Minus},
        //{"*", ttid::Mult},
        //{"/", ttid::Div},
        //{"(", ttid::Lparen},
        //{")", ttid::Rparen},
        //{",", ttid::Comma},
        //{".", ttid::Period},
        //{";", ttid::Semicolon},
        {"=", ttid::Equal},
        //{"<", ttid::Lss},
        //{">", ttid::Grtr},

        //{"<>", ttid::NotEq},
        //{"<=", ttid::LssEq},
        //{">=", ttid::GrtrEq},
        //{":=", ttid::Assign},
        //{"$dummy2",ttid::end_of_KeySym},
    };

    // 予約キーワード or Ident(ユーザー定義の識別子) を返す。
    TokenTypeId getTokenTypeId(std::string identName) {
        // https://cpprefjp.github.io/reference/map/map/find.html
        auto it = keywords.find(identName);
        if (it != keywords.end()) {
            return it->second;
        }

        return TokenTypeId::Ident;
    }

    std::unordered_map<TokenTypeId, std::string> ttid2str{
        {ttid::Begin, "begin"},
        {ttid::End, "end"},
        {ttid::If, "if"},
        {ttid::Then, "then"},
        {ttid::While, "while"},
        {ttid::Do, "do", },
        {ttid::Return, "return"},
        {ttid::Func, "function"},
        {ttid::Var, "var"},
        {ttid::Const, "const"},
        {ttid::Odd, "odd"},
        {ttid::Write, "write"},
        {ttid::WriteLn, "writeln"},
        {ttid::end_of_KeyWd, "end_of_KeyWd"},

        //　記号と名前(KeyId)の表　
        {ttid::Plus, "+"},
        {ttid::Minus, "-"},
        {ttid::Mult, "*"},
        {ttid::Div, "/"},
        {ttid::Lparen, "("},
        {ttid::Rparen, ")"},
        {ttid::Comma, ","},
        {ttid::Period, "."},
        {ttid::Semicolon, ";"},
        {ttid::Equal, "="},
        {ttid::Lss, "<"},
        {ttid::Grtr, ">"},

        {ttid::NotEq, "<>"},
        {ttid::LssEq, "<="},
        {ttid::GrtrEq, ">="},
        {ttid::Assign, ":="},
        {ttid::end_of_KeySym, "end_of_KeySym"},

        {ttid::Ident, "identifier"},
        {ttid::Num, "number"},
        {ttid::Illegal, "illegal"},
        {ttid::Ext, "Ext"},
        {ttid::end_of_Token, "end_of_Token"},

        {ttid::letter, "letter"},
        {ttid::digit, "digit"},
        {ttid::colon, "colon"},
        {ttid::others, "others"},
    };

    class Lexer {
    private:
        std::vector<std::string> _input;
        int _input_size;
        char _current_char;

        int _current_pos_line = -1;
        int _current_pos_column = -1;

        // input に一行ずつソースコードを格納。
        void _read_file(const std::string& file_path) {
            std::ifstream ifs(file_path);

            if (!ifs) {
                std::cerr << "failed to open" << std::endl;
                return;
            }

            std::string str;
            while (std::getline(ifs, str)) {
                _input.emplace_back(str.append("\n"));
            }
        }

        // プログラムとして意味を持たない文字をskip
        void _skip_meaningless_char() {
            auto is_meaningless = [&]() -> bool {
                if (_current_char == ' ') return true;
                if (_current_char == '\n') return true;
                if (_current_char == '\t') return true;
                if (_current_char == '\r') return true;

                return false;
                };

            while (is_meaningless()) {
                next_char();
            }
        }

    public:
        Lexer()
            : _current_char(' '), _input_size(0)
        {

        }

        Lexer(const std::string& file_path)
        {
            _read_file(file_path);
            _input_size = _input.size();
        }

        Lexer(const std::vector<std::string>& inputs)
        {
            for (const auto& line : inputs) {
                _input.push_back(line + "\n");
            }

            _current_char = ' ';
            _input_size = _input.size();
        }

        int get_input_size() const { return _input_size; }

        // 事前に一度先に呼んでおく
        void next_char() {
            // https://qiita.com/YukiMiyatake/items/8d10bca26246f4f7a9c8
            auto get_next_char = [&]() -> char {
                if (_current_pos_column == -1) {    // 行の先頭
                    if (++_current_pos_line >= _input_size) {
                        // EOF
                        // TODO : ファイルの終わりを知らせる方法
                        // HACK : '.' がしっかりと打たれていれば、ここの結果は何でも良い。
                        //        けれど、'.' が打たれていない場合は、ここで終点を示す記号'.' を返してやる。
                        // https://www.k-cube.co.jp/wakaba/server/ascii_code.html
                        return  EXT_CHAR;
                    }
                }

                char ch = _input[_current_pos_line][++_current_pos_column];
                if (ch == '\n') {
                    // 次の行へ移動
                    _current_pos_column = -1;
                }

                return ch;
                };

            _current_char = get_next_char();
        }

        Token next_token() {
            Token returnToken;
            _skip_meaningless_char();


            // 非LL(1) は個別に処理し、その他はdefault で処理。
            // LL(1) は構文解析の際にしか触れられていないが、この場合も似たような物。
            // あるTokenTypeId から始まる TokenTypeId が複数ある場合は、個別に対応する。
            /*
            開始Token    開始Tokenから始まるToken(これが複数なら非LL(1))
            .
            =
            ;
            ,
            (
            )
            +
            -
            *
            /                            // ここまでは、 開始Token のみ故省略
            :            :=
            <            < <> <=         // ここからは、複数の可能性がある。
            >            > >=
            */
            switch (TokenTypeId ccti = cppCharTypeId[_current_char]) {
            case TokenTypeId::letter:
                // TODO: 開始位置覚えておいて、一気にスライスで識別子名取得みたいなの出来ないかね？
                // HACK: 配列自体の代入が出来ないので、文字読み込むたびにreturnToken.u.name[i] に入れてる。
                //       本当はidentName[i] に入れておいて、最後にポインタで指してあげたい。
                //       けど、ローカル変数のポインタだからスコープ抜けると死ぬ。　　　
            {
                // 1字先読み有限オートマトン
                int i = 0;

                returnToken.u.name[i] = _current_char;
                ++i;

                next_char();
                while (
                    cppCharTypeId[_current_char] == TokenTypeId::letter
                    || cppCharTypeId[_current_char] == TokenTypeId::digit
                    ) {
                    returnToken.u.name[i] = _current_char;
                    ++i;

                    next_char();  // 最終的には1文字先読みすることになる。
                }
                returnToken.u.name[i] = '\0';    // 文字列の終端

                // TokenTypeId
                returnToken.typeId = getTokenTypeId(returnToken.u.name);
                break;
            }
            case TokenTypeId::digit:
            {
                // 1字先読み有限オートマトン
                int64_t num = _current_char - '0';

                next_char();

                while (cppCharTypeId[_current_char] == TokenTypeId::digit) {
                    num = num * 10 + (_current_char - '0');
                    next_char();  // 最終的には1文字先読みすることになる。
                }

                returnToken.typeId = TokenTypeId::Num;
                returnToken.u.value = num;
                break;
            }
            case TokenTypeId::colon:
                // 正しいプログラムのみが入力されるなら、不必要。だが、単体では意味を為さないので、
                // 後ろに'=' が来ない単体だった際にエラーを出力
            {
                // 1字先読み有限オートマトン
                next_char();
                if (_current_char == '=') {        // ":="
                    returnToken.typeId = TokenTypeId::Assign;
                    next_token();        // 先読み
                }
                else {
                    // この時、結果的に先頭のnext_char(); が先読みになっているので、先読み不要。
                    returnToken.typeId = TokenTypeId::Illegal;

                    // error
                    std::ostringstream oss;
                    oss << "IllegalTokenError:\n";
                    oss << "in Line = " << _current_pos_line << ", Column = " << _current_pos_column << ", ";
                    oss << "expected = ':=', but got = ':" << _current_char << "'";
                    g_error.pushMessage(oss.str());
                }
                break;
            }
            case TokenTypeId::Lss:
            {
                // 1字先読み有限オートマトン
                next_char();
                if (_current_char == '>') {        // "<>"
                    returnToken.typeId = TokenTypeId::NotEq;
                    next_token();        // 先読み
                }
                else if (_current_char == '=') {        // "<="
                    returnToken.typeId = TokenTypeId::LssEq;
                    next_token();        // 先読み
                }
                else {        // "<"
                    // この時、結果的に先頭のnext_char(); が先読みになっているので、先読み不要。
                    returnToken.typeId = TokenTypeId::Lss;
                }
                break;
            }
            case TokenTypeId::Grtr:
            {
                // 1字先読み有限オートマトン
                next_char();
                if (_current_char == '=') {        // ">="
                    returnToken.typeId = TokenTypeId::GrtrEq;
                    next_token();        // 先読み
                }
                else {        // ">"
                    // この時、結果的に先頭のnext_char(); が先読みになっているので、先読み不要。
                    returnToken.typeId = TokenTypeId::Grtr;
                }
                break;
            }
            default:
            {
                returnToken.typeId = ccti;
                next_char();        // 先読み
                break;
            }
            }
            return returnToken;
        }
    };

    class Entry {
    private:
        char name[MAXNAME];
        int value;

    public:
        Entry(const char _name[MAXNAME], const int _value)
            : value(_value)
        {
            set_name(_name);
        }

        std::string get_name() const { return name; }
        int get_value() const { return value; }

        
        void set_name(const std::string _name) { set_name(_name.c_str()); }
        void set_name(const char _name[MAXNAME]) { strcpy_s(name, _name); }
    };

    // 現状、一つのLLParser オブジェクトを使いまわすことは想定していない。(init() すれば動く可能性は勿論あるが非推奨。)
    // 1Token 先読みLL(1)構文解析器.
    // ・非終端記号解析関数の末尾は、必ず1Token 先読みされた状態で終わる。
    // ・非終端記号解析関数の呼出しは、必ず1Token 先読みされた状態で行う。
    class LLParser {
    private:
        Lexer _lexer;
        Token _current_token;

        std::vector<Entry> _name_table;    // これが実質的な生成される中間言語。

        // 次のToken を取得。
        inline void _read_next_token() {
            _current_token = _lexer.next_token();
        }

        inline bool _current_token_is(const TokenTypeId ttid) {
            if (_current_token.typeId == ttid) {
                return true;
            }
            return false;
        }

        std::string _unexpected_token_error_string(
            const std::string place_name, const TokenTypeId expect, const TokenTypeId got
        ) {
            std::ostringstream oss;
            oss
                << "" << place_name
                << " expected=" << ttid2str[expect]
                << ", but got=" << ttid2str[got];
            return oss.str();
        }

        // currentChar_ をcheck し、予期した通りなら先読みも行う。
        // 存在すること自体に意味があって、その中身に意味の無い奴なんかはこれでcheck だけして次に行く。
        bool _check_current_and_read_next(const TokenTypeId expected_ttid) {
            if (_current_token.typeId == expected_ttid) {    // 予期した通り
                _current_token = _lexer.next_token();
                return true;
            }

            std::ostringstream oss;
            oss << "Error: "
                << _unexpected_token_error_string("_check_current_and_read_next()", expected_ttid, _current_token.typeId);
            g_error.pushMessage(oss.str());
            return false;
        }

    public:
        LLParser(const std::string file_path) {
            _lexer = Lexer(file_path);
            init();
        }

        LLParser(const std::vector<std::string>& input) {
            _lexer = Lexer(input);
            init();
        }

        void init() {
            // nextToken_() を呼ぶ前に初期化する
            initcppCharTypeId();
            _lexer.next_char();    // 1token 先読み
            _read_next_token();    // lexer が先読みした奴をparser も貰う。

            _name_table.clear();

            g_error.init();
        }

        inline void compile() {
            bool isLoop = true;
            while (isLoop) {
                switch (_current_token.typeId) {
                case ttid::Ident:
                    const_decl();
                    break;
                    // ε に遷移したら何もしない。だって何もないから。ここでループを終了する。
                case ttid::Ext:
                    isLoop = false;
                    break;
                default:
                    isLoop = false;
                    g_error.pushMessage("Error: compile() got unexpected token=" + ttid2str[_current_token.typeId]);
                    break;
                }
            }
        }

        int get_input_size() const { return _lexer.get_input_size(); }

        // compile 後に呼ぶこと。
        std::vector<Entry> get_name_table() const { return _name_table; }
        void print_name_table() const {
            for (const auto& entry : _name_table) {
                std::cout << "[" << entry.get_name() << "," << entry.get_value() << "]" << std::endl;
            }
        }

        bool g_error_is_ok() const { return g_error.isOk(); }
        void g_error_flush() { g_error.flush(); }
        void g_error_flush_with_header(const std::string header) { g_error.flush_with_header(header); }

        // current_token = ident の状態で呼ぶ
        // constDecl は、以下の書き換え規則で表現される。
        //     constDec → ident eq Number
        //     Number   → Minus number | number
        inline void const_decl() {
            if (_current_token_is(ttid::Ident)) {
                Token ident_token = _current_token;
                _read_next_token();

                if (!_check_current_and_read_next(ttid::Equal)) {
                    g_error.pushMessage("parserError: in const_decl(), unexpected Token.");
                }

                if (_current_token_is(ttid::Num)) {
                    _name_table.emplace_back(ident_token.u.name, _current_token.u.value);
                }
                else if (_current_token_is(ttid::Minus)) {
                    _read_next_token();

                    if (_current_token_is(ttid::Num)) {
                        _name_table.emplace_back(ident_token.u.name, -_current_token.u.value);
                    }
                    else {
                        std::ostringstream oss;
                        oss << "Error: "
                            << _unexpected_token_error_string("const_decl()", ttid::Num, _current_token.typeId);
                        g_error.pushMessage(oss.str());
                    }
                }
                else {
                    std::ostringstream oss;
                    oss << "Error: "
                        << _unexpected_token_error_string("const_decl()", ttid::Num, _current_token.typeId);
                    g_error.pushMessage(oss.str());
                }

                // 先読みした状態で抜ける。
                _read_next_token();
            }
        }
    };
}
