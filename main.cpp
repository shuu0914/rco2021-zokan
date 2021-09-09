#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
// #define NDEBUG
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

int MAX_BUY_T = 800;
int SAKIYOMI_ERASE = 5;
int MAX_SAKIYOMI_DFS = 2;
int ADJ_PENA_THRESHOLD = 3;
int CENTER_ERASE_PENALTY = INF/2;

float CENTER_BONUS = 0.1;
float MAIN_MONEY_WEIGHT = 2.0;
float ADJ_PENALTY_WEIGHT = 1000.0;
int NOMUST_CONNECT_THRESHOLD = 3;
int SAKIYOMI_TURN = 3;

int START_SAKIYOMI = 500;

// int MAX_BUY_T = 844;
// int SAKIYOMI_ERASE = 7;
// int MAX_SAKIYOMI_DFS = 1;
// int ADJ_PENA_THRESHOLD = 4;
// int CENTER_ERASE_PENALTY = 10000;

// float CENTER_BONUS = 0.0756;
// float MAIN_MONEY_WEIGHT = 3.54;
// float ADJ_PENALTY_WEIGHT = 742.23;
// int NOMUST_CONNECT_THRESHOLD = 2;
// int SAKIYOMI_TURN = 0;

// int START_SAKIYOMI = 979;

const int MAX_HOHABA = 6;
const int BW = 10;

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

struct Dir{
    int y,x;

    bool operator==(const Dir& other) const{
        return y == other.y && x == other.x;
    }
};
constexpr Dir U = {-1, 0};
constexpr Dir D = {1, 0};
constexpr Dir L = {0, -1};
constexpr Dir R = {0, 1};
constexpr Dir NONE = {0, 0};

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
    Pos(){}

    Pos(int inp_y, int inp_x)
    : y(inp_y), x(inp_x) {}

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
        return {y + d.y, x + d.x};
    }

    Pos operator+ (const Pos& p) const{
        return {y + p.y, x + p.x};
    }

    void operator+= (Dir d){
        y += d.y;
        x += d.x;
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
        y -= d.y;
        x -= d.x;
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
vector<vector<int>> TP2S(T, vector<int>(N*N, -1));
vector<vector<int>> TP2V_ruiseki(T+1, vector<int>(N*N));
vector<vector<int>> TP2NS(T+1, vector<int>(N*N));
vector<vector<Pos>> T2P(T);
vector<vector<Event>> events(T+1), events_S(T+1);

int debug_final_money = 0;

enum ActionKind{
    BUY, MOVE, PASS
};

struct Action{
    ActionKind kind;
    Pos to, from;
};

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
    float eval = 0.0;
    array<int, N*N> last_takens;
    bitset<N*N> is_machines;
    vector<Pos> memo_machines;

    State(){
        fill(all(last_takens), -1);
        // fill(all(is_machines), false);
        is_machines = 0;
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
        return TP2V[t][p.idx()] > 0 && last_takens[p.idx()] < TP2S[t][p.idx()];
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

    void do_action(const Action& action){
        assert(can_action(action));
        if(action.kind == BUY){
            money -= get_cost();
            machine_count++;
            is_machines[action.to.idx()] = true;
            if(is_veg(action.to)){
                const int val = get_veg_value(action.to);
                money += val * count();
                eval -= val;
                last_takens[action.to.idx()] = t;
            }
        }else if(action.kind == MOVE){
            is_machines[action.from.idx()] = false;
            is_machines[action.to.idx()] = true;
            if(is_veg(action.to)){
                const int val = get_veg_value(action.to);
                money += val * count();
                eval -= val;
                last_takens[action.to.idx()] = t;
            }
        }else{
            //PASS
        }
        do_turn_end();
    }

    void do_turn_end(){
        for(const auto& event : events_S[t]){
            const Pos& p = event.p;
            if(!is_machine(p)) continue;
            if(last_takens[p.idx()] == t) continue;
            last_takens[p.idx()] = t;
            money += event.val * count();
            eval -= event.val;
        }
        t++;
    }

    const vector<Pos>& get_machines() const{
        return memo_machines;
    }

    void set_machines(vector<Pos>&& machines){
        memo_machines = std::forward<vector<Pos>>(machines);
    }

    void set_eval(const float& eval_){
        eval = eval_;
    }

    float evaluate() const{
        return eval + money;
    }

    bitset<N*N> hash() const{
        return is_machines;
    }
};

void dfs_kansetsu(const Pos& p, bitset<N*N>& checked, bitset<N*N>& is_machines, vector<Pos>& not_kansetsu_poses,
                  int& count, vector<int>& ord, vector<int>& low, const Dir before_dir){
    assert(p.in_range());
    ord[p.idx()] = count;
    count++;

    //根は追加しない
    bool is_kansetsu = ord[p.idx()] == 0;
    for(const auto& dir : DIRS4){
        if(~before_dir == dir) continue;
        const Pos&& pp = p + dir;
        if(!pp.in_range()) continue;
        if(!is_machines[pp.idx()]) continue;
        if(checked[pp.idx()]){
            //後退辺
            chmin(low[p.idx()], ord[pp.idx()]);
            continue;
        }
        checked[pp.idx()] = true;
        dfs_kansetsu(pp, checked, is_machines, not_kansetsu_poses, count, ord, low, dir);
        chmin(low[p.idx()], low[pp.idx()]);
        is_kansetsu = is_kansetsu || ord[p.idx()] <= low[pp.idx()];
    }
    if(!is_kansetsu){
        not_kansetsu_poses.emplace_back(p);
    }
}

vector<Pos> POSES_ALL;

template<typename Eval, class ValueEstimator>
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
        const auto push_action = [&](Action&& action, vector<Pos>&& next_machines, const Eval memo_eval, const int bonus = 0){
            State after_state = before_state;
            after_state.do_action(action);
            after_state.set_machines(std::forward<vector<Pos>>(next_machines));
            after_state.set_eval(memo_eval);
            const int t = after_state.turn();
            const Eval eval = after_state.evaluate() + bonus;
            vec_pq[t].emplace_back(eval, logs.size());
            logs.emplace_back(std::move(after_state), std::forward<Action>(action), before_idx, eval);
        };
        // const auto push_actions = [&](vector<Action>&& actions, State&& after_state, const int bonus = 0){
        //     const int t = after_state.turn();
        //     assert(t <= T);
        //     const Eval eval = after_state.evaluate() + bonus;
        //     vec_pq[t].emplace_back(eval, logs.size());
        //     logs.emplace_back(std::forward<State>(after_state), std::forward<vector<Action>>(actions), before_idx, eval);
        // };

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
            vector<Pos> next_machines;
            Action action;
            if(from.y == -1){
                action.kind = BUY;
                action.to = to;
                next_machines.emplace_back(to);
            }else{
                action.kind = MOVE;
                action.from = from;
                action.to = to;
                next_machines.emplace_back(to);
            }
            assert(before_state.can_action(action));
            push_action(std::move(action), std::move(next_machines), 0);
            return;
        }

        auto machines = before_state.get_machines();
        struct Mins{
            int first_min = INF;
            //Todo:高速化
            Pos first_argmin = {-1,-1};
            int second_min = INF;
            Pos second_argmin = {-1,-1};
        };
        vector<Mins> mins(N*N);
        vector<Pos> adj_points;
        vector<float> adj_evals;
        {
            //現在地、出発地
            queue<pair<Pos,Pos>> q;

            //machinesを用いてqueueの初期化
            for(const auto& p : machines){
                q.emplace(p, p);
                mins[p.idx()].first_min = 0;
                mins[p.idx()].first_argmin = p;
            }

            //bfsしてfirst_minとsecond_minを求めつつ隣接点を列挙する
            for(int d = 1; q.size() > 0; ++d){
                queue<pair<Pos,Pos>> next_q;
                while(q.size() > 0){
                    const auto p = q.front().first;
                    const auto base_p = q.front().second;
                    q.pop();
                    for(const auto& dir : DIRS4){
                        Pos&& pp = p + dir;
                        if(!pp.in_range()) continue;
                        if(mins[pp.idx()].first_argmin == base_p || mins[pp.idx()].second_argmin == base_p) continue;
                        if(mins[pp.idx()].first_min > d){
                            mins[pp.idx()].second_min = mins[pp.idx()].first_min;
                            mins[pp.idx()].second_argmin = mins[pp.idx()].first_argmin;
                            mins[pp.idx()].first_min = d;
                            mins[pp.idx()].first_argmin = base_p;
                            if(d == 1){
                                adj_points.emplace_back(pp);
                            }
                        }else if(mins[pp.idx()].second_min > d){
                            mins[pp.idx()].second_min = d;
                            mins[pp.idx()].second_argmin = base_p;
                        }else{
                            continue;
                        }
                        next_q.emplace(std::forward<Pos>(pp), base_p);
                    }
                }
                q = std::move(next_q);
            }
        }
        //各隣接点からbfsをして、隣接点の価値を求める
        vector<int> check_vec(N*N, -1);
        rep(i,adj_points.size()){
            const Pos& base_p = adj_points[i];
            check_vec[base_p.idx()] = i;
            assert(mins[base_p.idx()].first_min > 0);
            queue<Pos> q;
            q.push(base_p);
            float eval_sum = 0;
            for(int d = 0; q.size() > 0; ++d){
                queue<Pos> next_q;
                const auto p = q.front();
                q.pop();
                assert(d + 1 == mins[p.idx()].first_min);
                eval_sum += ValueEstimator::TPD(before_state, before_state.turn(), p, d) - ValueEstimator::TPD(before_state, before_state.turn(), p, d+1);
                for(const auto& dir : DIRS4){
                    Pos&& pp = p + dir;
                    if(!pp.in_range()) continue;
                    if(check_vec[pp.idx()] == i) continue;
                    if(d + 1 >= mins[pp.idx()].first_min) continue;
                    check_vec[pp.idx()] = i;
                    next_q.emplace(std::forward<Pos>(pp));
                }
                q = std::move(next_q);
            }
            assert(eval_sum >= 0);
            adj_evals.emplace_back(eval_sum);
        }
        assert(adj_evals.size() == adj_points.size());
        //最も評価値が良い隣接点を選択
        int best_i = 0;
        float best_eval = 0;
        rep(i,adj_evals.size()){
            const float eval = adj_evals[i];
            if(eval > best_eval){
                best_eval = eval;
                best_i = i;
            }
        }
        const Pos& best_add_pos = adj_points[best_i];
        //最も良い隣接点にmachineを追加し、bfsをしてfirst_min, second_minを更新する
        auto is_machines = before_state.is_machines;
        {
            const int i = N*N*N;
            const Pos& base_p = best_add_pos;
            assert(is_machines[base_p.idx()] == false);
            is_machines[base_p.idx()] = true;
            check_vec[base_p.idx()] = i;
            assert(mins[base_p.idx()].first_min > 0);
            queue<Pos> q;
            for(int d = 0; q.size() > 0; ++d){
                queue<Pos> next_q;
                const auto p = q.front();
                q.pop();
                if(mins[p.idx()].first_min > d){
                    mins[p.idx()].second_min = mins[p.idx()].first_min;
                    mins[p.idx()].second_argmin = mins[p.idx()].first_argmin;
                    mins[p.idx()].first_min = d;
                    mins[p.idx()].first_argmin = base_p;
                }else if(mins[p.idx()].second_min > d){
                    mins[p.idx()].second_min = d;
                    mins[p.idx()].second_argmin = base_p;
                }
                for(const auto& dir : DIRS4){
                    Pos&& pp = p + dir;
                    if(!pp.in_range()) continue;
                    if(check_vec[pp.idx()] == i) continue;
                    if(d + 1 >= mins[pp.idx()].second_min) continue;
                    check_vec[pp.idx()] = i;
                    next_q.emplace(std::forward<Pos>(pp));
                }
                q = std::move(next_q);
            }
        }
        //全マスを走査して評価値を求めつつ、各点を取り除いた時の減少量を求める
        float total_eval = 0;
        //正の値
        vector<float> dec_evals(N*N,0);
        for(const auto& p : POSES_ALL){
            total_eval += ValueEstimator::TPD(before_state, before_state.turn(), p, mins[p.idx()].first_min);
            if(mins[p.idx()].first_min == mins[p.idx()].second_min) continue;
            dec_evals[mins[p.idx()].first_argmin.idx()] += ValueEstimator::TPD(before_state, before_state.turn(), p, mins[p.idx()].first_min);
            dec_evals[mins[p.idx()].first_argmin.idx()] -= ValueEstimator::TPD(before_state, before_state.turn(), p, mins[p.idx()].second_min);
            assert(dec_evals[mins[p.idx()].first_argmin.idx()] >= 0);
        }
        //BUY可能ならBUY
        if(before_state.can_buy()){
            Action action;
            action.kind = BUY;
            action.to = best_add_pos;
            machines.emplace_back(best_add_pos);
            push_action(std::move(action), std::move(machines), total_eval);
            return;
        }
        //追加した点からdfsをし、非関節点を列挙する
        bitset<N*N> checked = false;
        checked[best_add_pos.idx()] = true;
        int count = 0;
        vector<int> ord(N*N), low(N*N,INF);
        vector<Pos> not_kansetsu_poses;
        dfs_kansetsu(best_add_pos, checked, is_machines, not_kansetsu_poses, count, ord, low, NONE);
        assert(not_kansetsu_poses.size() > 0);

        //最も良い非関節点を取り除き、評価値を更新する
        float best_dec = INF;
        int best_i2 = 0;
        rep(i,not_kansetsu_poses.size()){
            const Pos& p = not_kansetsu_poses[i];
            const float dec = dec_evals[p.idx()];
            if(dec < best_dec){
                best_dec = dec;
                best_i2 = i;
            }
        }
        const Pos& best_remove_pos = not_kansetsu_poses[best_i2];
        total_eval -= best_dec;
        //遷移
        machines.erase(find(all(machines), best_remove_pos));
        machines.emplace_back(best_add_pos);
        Action action;
        action.kind = MOVE;
        action.from = best_remove_pos;
        action.to = best_add_pos;
        push_action(std::move(action), std::move(machines), total_eval);
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
            cerr<<t<<endl;
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
    rep(t,T+1){
        fill(all(TP2NS[t]),INF);
    }
    rep(i,M){
        int r,c,s,e,v;cin>>r>>c>>s>>e>>v;
        e++;
        V.push_back({r,c,s,e,v});
        for(int t = s; t < e; ++t){
            TP2V[t][idx(r,c)] += v;
            TP2S[t][idx(r,c)] = s;
            T2P[t].push_back({r,c});
        }
        if(s > 0){
            TP2NS[s-1][idx(r,c)] = s;
        }
        TP2V_ruiseki[s][idx(r,c)] += v;

        {
            Event event;
            event.p = {r,c};
            event.val = v;
            event.is_S = true;
            events[s].push_back(event);
            events_S[s].push_back(event);
            event.is_S = false;
            events[e].push_back(event);
        }
    }
    rep(idx,N*N){
        int s = INF;
        REP(t,T+1){
            chmin(TP2NS[t][idx], s);
            s = TP2NS[t][idx];
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

template<class ValueEstimator>
pair<vector<Action>, int> solve(){
    State first_state_;
    first_state_.money = 1;
    BeamSearcher<int, ValueEstimator> bs_er(first_state_);
    auto&& ans = bs_er.solve();
    cerr<<timer.ms()<<"[ms]"<<endl;
    const int final_money = debug_final_money;
    cerr<<"final money:"<<final_money<<endl;
    return std::make_pair(std::forward<decltype(ans)>(ans), final_money);
}

struct Estimator{
    static float TPD(const State& state, const int t, const Pos& p, const int d){
        if(d > 0) return 0;
        return state.get_veg_value(p);
    }
};

int main(int argc, char *argv[]){
    fast_io;

    if(argc >= 2){
        MAX_BUY_T = stoi(argv[1]);
        SAKIYOMI_ERASE = stoi(argv[2]);
        MAX_SAKIYOMI_DFS = stoi(argv[3]);
        ADJ_PENA_THRESHOLD = stoi(argv[4]);
        CENTER_ERASE_PENALTY = stoi(argv[5]);
        CENTER_BONUS = stof(argv[6]);
        MAIN_MONEY_WEIGHT = stof(argv[7]);
        ADJ_PENALTY_WEIGHT = stof(argv[8]);
        NOMUST_CONNECT_THRESHOLD = stoi(argv[9]);
        SAKIYOMI_TURN = stoi(argv[10]);
        START_SAKIYOMI = stoi(argv[11]);
    }

    input();

    const auto& pa = solve<Estimator>();
    cout<<pa.first<<endl;
    cerr<<"score:"<<pa.second*50<<endl;
    return 0;
}
