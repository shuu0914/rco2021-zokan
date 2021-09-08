#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#define NDEBUG
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <math.h>
#include <iomanip>
#include <limits>
#include <list>
#include <queue>
#include <tuple>
#include <map>
#include <stack>
#include <set>
#include <unordered_set>
#include <bitset>
#include <functional>
#include <chrono>
#include <cassert>
#include <array>
#ifdef PERF
#include <gperftools/profiler.h>
#endif
using namespace std;
#define fast_io ios_base::sync_with_stdio (false) ; cin.tie(0) ; cout.tie(0) ;
#define ll long long int
#define rep(i,n) for(int i=0; i<(int)(n); i++)
#define reps(i,n) for(int i=1; i<=(int)(n); i++)
#define REP(i,n) for(int i=n-1; i>=0; i--)
#define REPS(i,n) for(int i=n; i>0; i--)
#define MOD (long long int)(1e9+7)
#define INF (int)(1e9)
#define LINF (long long int)(1e18)
#define chmax(a, b) a = (((a)<(b)) ? (b) : (a))
#define chmin(a, b) a = (((a)>(b)) ? (b) : (a))
#define all(v) v.begin(), v.end()
typedef pair<int, int> Pii;
typedef pair<ll, ll> Pll;

uint32_t xorshift(){
    static uint32_t x = 123456789;
    static uint32_t y = 362436069;
    static uint32_t z = 521288629;
    static uint32_t w = 88675123;
    uint32_t t;
    t = x ^ (x<<11);
    x = y; y = z; z = w;
    w ^= t ^ (t>>8) ^ (w>>19);
    return w;
}

class Timer{
    chrono::system_clock::time_point _start, _end;
    ll _sum = 0, _count = 0;

public:
    void start(){
        _start = chrono::system_clock::now();
    }

    void stop(){
        _end = chrono::system_clock::now();
    }

    void add(){
        const chrono::system_clock::time_point now = chrono::system_clock::now();
        _sum += static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(now - _start).count());
        _count++;
    }

    ll sum(){
        return _sum / 1000;
    }

    string average(){
        if(_count == 0){
            return "NaN";
        }
        return to_string(_sum / 1000 / _count);
    }

    void reset(){
        _start = chrono::system_clock::now();
        _sum = 0;
        _count = 0;
    }

    inline int ms() const{
        const chrono::system_clock::time_point now = chrono::system_clock::now();
        return static_cast<double>(chrono::duration_cast<chrono::microseconds>(now - _start).count() / 1000);
    }

    inline int ns() const{
        const chrono::system_clock::time_point now = chrono::system_clock::now();
        return static_cast<double>(chrono::duration_cast<chrono::microseconds>(now - _start).count());
    }
};

Timer timer, timer1, timer2, timer3, timer4, timer5;

enum Dir{
    U,D,L,R,NONE
};

Dir operator~(const Dir& d){
    if(d == U){
        return D;
    }else if(d == D){
        return U;
    }else if(d == L){
        return R;
    }else if(d == R){
        return L;
    }else{
        return NONE;
    }
}

ostream& operator<<(ostream& os, const Dir& d){
    if(d == U){
        os<<"U";
    }else if(d == D){
        os<<"D";
    }else if(d == L){
        os<<"L";
    }else if(d == R){
        os<<"R";
    }else{
        os<<"-";
    }
    return os;
}

ostream& operator<<(ostream& os, const vector<Dir>& dirs){
    for(auto&& dir : dirs){
        os<<dir;
    }
    return os;
}

constexpr Dir DIRS4[4] = {U, D, L, R};

constexpr int N = 16, M = 5000, T = 1000;
struct Pos{
    int y,x;
    Pos(){
        y = -1;
        x = -1;
    }
    Pos(int inp_y, int inp_x){
        y = inp_y;
        x = inp_x;
    }

    void set_y(const int y_){
        y = y_;
    }

    void set_x(const int x_){
        x = x_;
    }

    int manhattan(const Pos& p) const{
        return abs(p.y - y) + abs(p.x - x);
    }

    int chebyshev(const Pos& p) const{
        return max(abs(p.x - x), abs(p.y - y));
    }

    Pos operator+ (Dir d) const{
        if(d == U){
            return {y-1, x};
        }else if(d == D){
            return {y+1, x};
        }else if(d == L){
            return {y, x-1};
        }else if(d == R){
            return {y, x+1};
        }else if(d == NONE){
            return {y, x};
        }

        throw invalid_argument("Posに加えるDirが適切ではありません");
    }

    Pos operator+ (const Pos& p) const{
        return {y + p.y, x + p.x};
    }

    void operator+= (Dir d){
        if(d == U){
            y -= 1;
        }else if(d == D){
            y += 1;
        }else if(d == L){
            x -= 1;
        }else if(d == R){
            x += 1;
        }else if(d == NONE){

        }else{
            throw invalid_argument("Posに+=するDirが適切ではありません");
        }
    }

    Dir operator- (const Pos& from) const{
        if(y == from.y){
            if(from.x + 1 == x){
                return R;
            }else if(from.x - 1 == x){
                return L;
            }
        }else if(x == from.x){
            if(from.y + 1 == y){
                return D;
            }else if(from.y - 1 == y){
                return U;
            }
        }
        throw invalid_argument("Posに-するPosが適切ではありません");
    }

    void operator-= (Dir d){
        if(d == D){
            y -= 1;
        }else if(d == U){
            y += 1;
        }else if(d == R){
            x -= 1;
        }else if(d == L){
            x += 1;
        }else if(d == NONE){

        }else{
            throw invalid_argument("Posに-=するDirが適切ではありません");
        }
    }

    bool operator==(const Pos& p) const{
        return x == p.x && y == p.y;
    }

    bool operator!=(const Pos& p) const{
        return x != p.x || y != p.y;
    }

    //mapに突っ込めるようにするために定義
    bool operator<(const Pos& p) const{
        if(y != p.y){
            return y < p.y;
        }
        return x < p.x;
    }

    bool in_range() const{
        return x >= 0 && x < N && y >= 0 && y < N;
    }

    string to_string() const{
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }

    friend ostream& operator<<(ostream& os, const Pos& p){
        os << "(" << p.y << "," << p.x << ")";
        return os;
    }

    string to_answer() const{
        return std::to_string(y) + " " + std::to_string(x);
    }

    int idx() const{
        assert(y != -1);
        return y * N + x;
    }
};

int idx(const int y, const int x){
    return y * N + x;
}

struct Veg{
    int r,c,s,e,v;
};
vector<Veg> V;

struct Event{
    bool is_S;
    Pos p;
    int val;
};

vector<vector<int>> TP2V(T, vector<int>(N*N));
vector<vector<int>> TP2V_ruiseki(T+1, vector<int>(N*N));
vector<vector<Pos>> T2P(T);
vector<vector<Event>> events(T+1);

int debug_final_money = 0;

enum ActionKind{
    BUY, MOVE, PASS
};

struct Action{
    ActionKind kind;
    Pos to, from;
};

constexpr int MAX_BUY_T = 800;
constexpr int MAX_HOHABA = 6;
const int BW = 10;

ostream& operator<<(ostream& os, const vector<Action>& actions){
    rep(i, actions.size()){
        const auto& action = actions[i];
        if(action.kind == BUY){
            os << action.to.to_answer();
        }else if(action.kind == MOVE){
            os << action.from.to_answer() << " " << action.to.to_answer();
        }else{
            os << "-1";
        }
        if(i+1 != actions.size()){
            os << endl;
        }
    }
    return os;
}

struct State{
    int money = 1;
    int t = 0;
    int machine_count = 0;
    bitset<N*N> is_machines = 0;
    bitset<N*N> is_vegs = 0;
    bitset<N*N> is_kansetsu_ = 0;
    int reserve_money = 0;
    int max_connect_count = 0;
    int adj_pena = 0;
    // int min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    // int mitsu_pena = 0;

    State(){
        for(const auto& p : T2P[0]){
            is_vegs[p.idx()] = true;
        }
    }

    int get_cost() const{
        return (machine_count + 1) * (machine_count + 1) * (machine_count + 1);
    }

    bool can_buy() const{
        return get_cost() <= money;
    }

    bool is_machine(const Pos& p) const{
        return is_machines[p.idx()];
    }

    bool is_veg(const Pos& p) const{
        return is_vegs[p.idx()];
    }

    int get_veg_value(const Pos& p) const{
        if(!is_veg(p)){
            return 0;
        }
        return TP2V[t][p.idx()];
    }

    int count() const{
        return machine_count;
    }

    int turn() const{
        return t;
    }

    int get_money() const{
        return money;
    }

    vector<Pos> get_machines() const{
        vector<Pos> ret;
        ret.reserve(machine_count);
        rep(y,N){
            rep(x,N){
                const Pos&& p = {y,x};
                if(!is_machine(p)) continue;
                ret.emplace_back(p);
            }
        }
        return ret;
    }

    int count_adj_machine(const Pos& p) const{
        int ret = 0;
        for(const auto& dir : DIRS4){
            const Pos&& pp = p + dir;
            if(!pp.in_range()) continue;
            ret += is_machine(pp);
        }
        return ret;
    }

    bool can_action(const Action& action) const{
        const auto adj_count = [&](const Pos& p){
            int count = 0;
            for(const auto& dir : DIRS4){
                const Pos&& pp = p + dir;
                if(!pp.in_range()) continue;
                count += is_machine(pp);
            }
            return count;
        };
        if(action.kind == BUY){
            if(!can_buy()){
                return false;
            }
            if(is_machine(action.to)){
                return false;
            }
            //Todo:大丈夫？
            // if(adj_count(action.to) >= 2){
            //     return false;
            // }
            return true;
        }else if(action.kind == MOVE){
            if(!is_machine(action.from)){
                return false;
            }
            if(is_machine(action.to)){
                return false;
            }
            //Todo:大丈夫？
            // if(adj_count(action.to) >= 2){
            //     return false;
            // }
            return true;
        }else{
            //PASS
            return true;
        }
    }

    bool is_kansetsu(const Pos& p) const{
        return is_kansetsu_[p.idx()];
    }

    void do_action(const Action& action, const vector<Pos>& machines){
        assert(can_action(action));
        if(action.kind == BUY){
            money -= get_cost();
            machine_count++;
            is_machines[action.to.idx()] = true;
        }else if(action.kind == MOVE){
            is_machines[action.from.idx()] = false;
            is_machines[action.to.idx()] = true;
        }else{
            //PASS
        }
        do_turn_end(machines);
    }

    void do_action(const Action& action){
        assert(can_action(action));
        if(action.kind == BUY){
            money -= get_cost();
            machine_count++;
            is_machines[action.to.idx()] = true;
        }else if(action.kind == MOVE){
            is_machines[action.from.idx()] = false;
            is_machines[action.to.idx()] = true;
        }else{
            //PASS
        }
        do_turn_end(get_machines());
    }

    void dfs(const Pos& p, bitset<N*N>& checked, int& count, int& sum_val, int& sum_reserve_val, vector<int>& ord, vector<int>& low){
        const bool is_root = count == 0;
        assert(p.in_range());
        // chmin(min_x, p.x);
        // chmax(max_x, p.x);
        // chmin(min_y, p.y);
        // chmax(max_y, p.y);
        ord[p.idx()] = count;
        count++;
        int root_count = 0;
        if(is_veg(p)){
            sum_val += TP2V[t][p.idx()];
            is_vegs[p.idx()] = false;
        }
        //Todo:コアレス
        const int sakiyomi = min(10, this->count() - 1);
        sum_reserve_val += TP2V_ruiseki[min(T, t+sakiyomi)][p.idx()] - TP2V_ruiseki[t][p.idx()];
        int adj_count = 0;
        for(const auto& dir : DIRS4){
            const Pos pp = p + dir;
            if(!pp.in_range()) continue;
            if(!is_machine(pp)) continue;
            adj_count++;
            if(checked[pp.idx()]){
                //後退辺
                chmin(low[p.idx()], ord[pp.idx()]);
                continue;
            }
            root_count++;
            checked[pp.idx()] = true;
            dfs(pp, checked, count, sum_val, sum_reserve_val, ord, low);
            chmin(low[p.idx()], low[pp.idx()]);
            if(!is_root && ord[p.idx()] <= low[pp.idx()]){
                is_kansetsu_[p.idx()] = true;
            }
        }

        // int adj8_count = adj_count;
        // for(auto&& dy : {-1,1}){
        //     for(auto&& dx : {-1,1}){
        //         const Pos&& pp = p + Pos(dy,dx);
        //         if(!pp.in_range()) continue;
        //         adj8_count += is_machine(pp);
        //     }
        // }
        // if(adj8_count >= 4){
            // mitsu_pena += 1;
        // }

        if(adj_count >= 3){
            adj_pena++;
        }
        //Todo:根の関節点判定にバグないかチェック
        if(is_root && root_count >= 2){
            is_kansetsu_[p.idx()] = true;
        }
    }

    void do_turn_end(const vector<Pos>& machines){
        bitset<N*N> checked = 0;
        reserve_money = 0;
        max_connect_count = 0;
        adj_pena = 0;
        // min_x = INF;
        // max_x = 0;
        // min_y = INF;
        // max_y = 0;
        // mitsu_pena = 0;

        vector<int> ord(N*N), low(N*N, INF);
        is_kansetsu_ = 0;

        for(const auto& base_p : machines){
            if(checked[base_p.idx()]) continue;
            checked[base_p.idx()] = true;
            int count = 0;
            int sum_val = 0;
            int sum_reserve_val = 0;
            dfs(base_p, checked, count, sum_val, sum_reserve_val, ord, low);
            money += count * sum_val;
            reserve_money += count * sum_reserve_val;
            chmax(max_connect_count, count);
        }
        t++;

        //Todo:評価値悪かったらStart処理はする必要ない
        for(const auto& event : events[t]){
            const Pos& p = event.p;
            assert(p.in_range());
            if(event.is_S){
                is_vegs[p.idx()] = true;
            }else{
                is_vegs[p.idx()] = false;
            }
        }
    }

    int evaluate() const{
        int eval = 0;
        eval += count() * (t < MAX_BUY_T ? 1e9 / (N*N) : 0);
        eval += money * 2;
        eval += reserve_money;
        // eval += max_connect_count * max_connect_count;
        // if(count() >= 4 && max_connect_count != count()){
        //     eval = -1;
        // }
        eval -= adj_pena * 1000;
        // eval += ((max_x - min_x) + (max_y - min_y)) * count();
        // eval -= mitsu_pena * mitsu_pena * mitsu_pena * get_cost() * 0.01;
        return eval;
    }

    bitset<N*N> hash() const{
        return is_vegs ^ is_machines;
    }
};

vector<Pos> POSES_ALL;

template<typename Eval>
struct BeamSearcher{
    struct Log{
        State state;
        Action action;
        vector<Action> actions;
        int before_idx;
        Eval eval;

        Log(){

        }

        Log(State&& state_, Action&& action_, const int before_idx_, const Eval eval_)
        : state(std::forward<State>(state_)), action(std::forward<Action>(action_)), before_idx(before_idx_), eval(eval_)
        {

        }

        Log(State&& state_, const Action& action_, const int before_idx_, const Eval eval_)
        : state(std::forward<State>(state_)), action(action_), before_idx(before_idx_), eval(eval_)
        {

        }

        Log(State&& state_, vector<Action>&& actions_, const int before_idx_, const Eval eval_)
        : state(std::forward<State>(state_)), actions(std::forward<vector<Action>>(actions_)), before_idx(before_idx_), eval(eval_)
        {

        }

        Log(State&& state_, const vector<Action>& actions_, const int before_idx_, const Eval eval_)
        : state(std::forward<State>(state_)), actions(actions_), before_idx(before_idx_), eval(eval_)
        {

        }

        void set_state(const State& inp_state){
            state = inp_state;
        }

        bool operator<(const Log& obj) const{
            return eval < obj.eval;
        }
    };

    State first_state;
    vector<Log> logs;
    vector<vector<pair<Eval, int>>> vec_pq;

    BeamSearcher(const State& first_state_)
    : first_state(first_state_)
    {
    }

    void expand(const State& before_state, const int before_idx){
        const auto push_action = [&](Action&& action, const int bonus = 0){
            State after_state = before_state;
            after_state.do_action(action);
            const int t = after_state.turn();
            const Eval eval = after_state.evaluate() + bonus;
            vec_pq[t].emplace_back(eval, logs.size());
            logs.emplace_back(std::move(after_state), std::forward<Action>(action), before_idx, eval);
        };
        const auto push_actions = [&](vector<Action>&& actions, State&& after_state, const int bonus = 0){
            const int t = after_state.turn();
            assert(t <= T);
            const Eval eval = after_state.evaluate() + bonus;
            vec_pq[t].emplace_back(eval, logs.size());
            logs.emplace_back(std::forward<State>(after_state), std::forward<vector<Action>>(actions), before_idx, eval);
        };

        const auto before_machines = before_state.get_machines();

        if(before_state.count() == 0 || (before_state.count() == 1 && !before_state.can_buy())){
            Pos from = {-1,-1};
            int max_val = 0;
            Pos to = {-1,-1};
            for(const Pos& p : POSES_ALL){
                if(before_state.is_machine(p)){
                    from = p;
                }
                const int val = before_state.get_veg_value(p);
                if(val > max_val){
                    max_val = val;
                    to = p;
                }
            }
            assert(to.y != -1);

            Action action;
            if(from.y == -1){
                action.kind = BUY;
                action.to = to;
            }else{
                action.kind = MOVE;
                action.from = from;
                action.to = to;
            }
            assert(before_state.can_action(action));
            push_action(std::move(action));
            return;
        }

        const auto func = [&](const auto UD, const auto LR, const bool must_connect = true){
            vector<int> dp(N*N);
            if(must_connect){
                for(const auto& p : POSES_ALL){
                    if(!before_state.is_machine(p)){
                        dp[p.idx()] = -INF;
                    }
                }
            }
            //Todo:正しい？
            const int HOHABA = MAX_HOHABA;
            vector<vector<Pos>> before_pos(HOHABA, vector<Pos>(N*N));
            vector<int> vec_max_val;
            vector<vector<Pos>> vec_max_keiro;
            rep(_t, HOHABA){
                //Todo:あってる？
                const int t = before_state.turn() + _t;
                if(t >= T) break;
                int max_val = -1;
                Pos max_pos = {-1,-1};
                vector<int> dp2(N*N,-INF);
                for(const auto& p : POSES_ALL){
                    if(before_state.is_machine(p)) continue;
                    //Todo:取得済みかどうかのチェック
                    //Todo:累積のほうが良いかも？
                    const int val = [&](){
                        if(t < 500 || before_state.turn() + 3 <= t){
                            //connectしていない場合は何歩目かによって価値が変わる
                            return TP2V[t][p.idx()] * (must_connect ? 1 : _t + 1);
                        }else{
                            //Todo:先読みターン数
                            //Todo:提出時にはassert外すかNDEBUG
                            //Todo:must_connectではないときも先読みしたい 3が降ってくる前において、その後隣に置くことで3*2点をしたい
                            assert(must_connect);
                            return TP2V_ruiseki[min(T, before_state.turn() + 3)][p.idx()] - TP2V_ruiseki[t][p.idx()] + TP2V[t][p.idx()];
                        }
                    }();

                    const Pos&& pp1 = p + UD;
                    if(pp1.in_range() && dp2[p.idx()] < dp[pp1.idx()] + val){
                        dp2[p.idx()] = dp[pp1.idx()] + val;
                        before_pos[_t][p.idx()] = pp1;
                    }

                    const Pos&& pp2 = p + LR;
                    if(pp2.in_range() && dp2[p.idx()] < dp[pp2.idx()] + val){
                        dp2[p.idx()] = dp[pp2.idx()] + val;
                        before_pos[_t][p.idx()] = pp2;
                    }

                    if(dp2[p.idx()] > max_val){
                        max_val = dp2[p.idx()];
                        max_pos = p;
                    }
                }

                dp = std::move(dp2);

                if(max_val == -1) continue;

                if(!must_connect && _t+1 < before_state.count()) continue;

                vec_max_val.emplace_back(max_val);
                Pos p = max_pos;
                assert(p.y != -1);
                vector<Pos> kei;
                REP(_i, _t+1){
                    kei.emplace_back(p);
                    p = before_pos[_i][p.idx()];
                }
                reverse(all(kei));
                vec_max_keiro.emplace_back(kei);
            }

            assert(vec_max_val.size() == vec_max_keiro.size());

            // 最初の一歩から最も遠いものから順に消していく
            // Todo:最適化
            // Todo:複数手
            rep(i, vec_max_keiro.size()){
                const auto& keiro = vec_max_keiro[i];
                //遠いものから
                //Todo:消すものが足りない場合に追加したものをそのまま消す可能性
                //先頭は一歩目なので無視

                vector<Action> actions;
                State after_state = before_state;
                for(const auto& to : keiro){
                    Action action;
                    //Todo:複数回購入できる可能性
                    if(after_state.turn() <= MAX_BUY_T && after_state.can_buy()){
                        action.kind = BUY;
                        action.to = to;
                    }else{
                        const auto evaluate = [&](const Pos& from){
                            if(from.manhattan(to) == 1) return -INF;
                            constexpr int sakiyomi = 5;
                            const int t = after_state.turn();
                            const int saki_t = min(t, T);
                            return -(TP2V_ruiseki[saki_t][from.idx()] - TP2V_ruiseki[t][from.idx()]);
                        };
                        Pos best_from = {-1,-1};
                        int best_eval = -INF;
                        //must_connectでないときでも、before_machinesの中から関節点ではないものを1つずつ消していくのでこれでよい
                        //Todo:序盤は関節点を削除したほうが評価値が向上する可能性
                        for(const auto& from : before_machines){
                            if(!after_state.is_machine(from)) continue;
                            if(after_state.is_kansetsu(from)) continue;
                            const int eval = evaluate(from);
                            if(eval > best_eval){
                                best_eval = eval;
                                best_from = from;
                            }
                        }
                        if(best_from.y == -1){
                            break;
                        }
                        action.kind = MOVE;
                        action.from = best_from;
                        action.to = to;
                    }

                    // cerr<<action.kind<<" "<<action.from<<" "<<action.to<<" "<<keiro.size()<<" "<<vec_bfs.size()<<endl;
                    // assert(before_state.can_action(action));
                    actions.emplace_back(action);
                    assert(after_state.can_action(action));
                    after_state.do_action(action);
                    //Todo:2手以上を突っ込む方法
                    // assert(HOHABA == 1);
                    //Todo:breakしない
                    // break;
                }
                push_actions(std::move(actions), std::move(after_state));
            }
        };

        for(const auto UD : {U,D}){
            for(const auto LR : {L,R}){
                func(UD, LR);
            }
        }
        if(before_state.count() <= 3){
            for(const auto UD : {U,D}){
                for(const auto LR : {L,R}){
                    func(UD, LR, false);
                }
            }
        }
    }

    vector<Action> back_prop(const int last_idx){
        debug_final_money = logs[last_idx].state.get_money();
        vector<Action> ans;
        int idx = last_idx;
        while(idx != 0){
            if(logs[idx].actions.size() == 0){
                ans.emplace_back(logs[idx].action);
            }else{
                REP(j,logs[idx].actions.size()){
                    const auto& action = logs[idx].actions[j];
                    ans.emplace_back(action);
                }
            }
            idx = logs[idx].before_idx;
        }
        reverse(all(ans));
        return ans;
    }

    vector<Action> solve(){
        vec_pq.resize(T+1);
        rep(i,T+1){
            //4は{UD}*{LR}の組み合わせ数
            //2はmust_connectかどうかで2通り
            vec_pq[i].reserve(BW * MAX_HOHABA * 4 * 2);
        }
        logs.reserve((int)1e7);
        {
            Log first_log;
            first_log.set_state(first_state);
            logs.emplace_back(first_log);
            assert(logs.size() == 1);
            vec_pq[0].emplace_back(0, 0);
        }
        rep(t, T){
            auto& current_pq = vec_pq[t];
            // partial_sort(current_pq.begin(), current_pq.begin() + min(BW * 2, (int)current_pq.size()), current_pq.end(), greater<>());
            sort(all(current_pq), greater<>());
            int vec_idx = 0;
            unordered_set<bitset<N*N>> S;
            for(int _t = 0; _t < BW && vec_idx < current_pq.size(); ++_t, ++vec_idx){
                const int idx = current_pq[vec_idx].second;
                const auto& state = logs[idx].state;
                const auto& hash = state.hash();
                if(S.count(hash) > 0){
                    _t--;
                    continue;
                }
                S.insert(hash);

                expand(state, idx);
            }
        }

        const auto& final_pq = vec_pq[T];
        const auto itr = max_element(all(final_pq));
        const vector<Action> ans = back_prop(itr->second);
        // State state;
        // for(const auto& action : ans){
        //     state.do_action(action);
        //     cerr<<state.t-1<<" "<<state.money<<endl;
        // }
        return ans;
    }
};

void input(){
    int _; cin>>_>>_>>_;
    timer.start();
    rep(i,M){
        int r,c,s,e,v;cin>>r>>c>>s>>e>>v;
        e++;
        V.push_back({r,c,s,e,v});
        for(int t = s; t < e; ++t){
            TP2V[t][idx(r,c)] += v;
            T2P[t].push_back({r,c});
        }
        TP2V_ruiseki[s][idx(r,c)] += v;

        {
            Event event;
            event.p = {r,c};
            event.val = v;
            event.is_S = true;
            events[s].push_back(event);
            event.is_S = false;
            events[e].push_back(event);
        }
    }
    rep(idx, N*N){
        rep(t,T){
            TP2V_ruiseki[t+1][idx] += TP2V_ruiseki[t][idx];
        }
    }
    rep(t,T+1){
        sort(all(events[t]),[&](const Event& l, const Event& r){
            if(l.is_S != r.is_S){
                //rがSならlはEで、lのほうが左
                return r.is_S;
            }
            //なんでもよい
            return l.val < r.val;
        });
    }
    rep(y,N){
        rep(x,N){
            POSES_ALL.push_back({y,x});
        }
    }
}

void solve(){
    State first_state_;
    first_state_.money = 1;
    BeamSearcher<int> bs_er(first_state_);
    cout<<bs_er.solve()<<endl;
    cerr<<timer.ms()<<"[ms]"<<endl;
    cerr<<"final money:"<<debug_final_money<<endl;
    cerr<<"score:"<<debug_final_money * 50<<endl;
}

int main(void){
    fast_io;

    input();
    solve();
    return 0;
}