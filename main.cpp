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

int MAX_BUY_COUNT = 50;
int NOMUST_CONNECT_THRESHOLD = 3;
int START_SAKIYOMI = 200;

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
vector<vector<int>> TP2S(T, vector<int>(N*N));
// vector<vector<int>> TP2V_ruiseki(T+1, vector<int>(N*N));
// vector<vector<int>> TP2NS(T+1, vector<int>(N*N));
vector<vector<float>> TP2eval(T, vector<float>(N*N));
vector<vector<Pos>> T2P(T);
vector<vector<Event>> events(T+1);

int debug_final_money = 0;

vector<uint16_t> checked(N*N,0);
vector<uint16_t> ord(N*N), low(N*N,0xffff);
uint16_t ord_root = (uint16_t)0 - N*N - 1;
uint16_t check_num = 1;

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

vector<Pos> POSES_ALL;
vector<vector<Pos>> POSES_EDGE(N*N);

template<class CenterJudger>
struct State_tmp{
    int money = 1;
    int t = 0;
    int machine_count = 0;
    bitset<N*N> is_machines = 0;
    bitset<N*N> is_vegs = 0;
    bitset<N*N> is_kansetsu_ = 0;
    float reserve_money = 0;
    int max_connect_count = 0;

    State_tmp(){
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

    void change_machine(const Pos& p, const bool bl){
        assert(is_machine(p) != bl);
        is_machines[p.idx()] = bl;
    }

    vector<Pos> get_machines() const{
        vector<Pos> ret;
        ret.reserve(machine_count);
        rep(y,N){
            rep(x,N){
                Pos&& p = {y,x};
                if(!is_machine(p)) continue;
                ret.emplace_back(std::forward<Pos>(p));
            }
        }
        return ret;
    }

    int count_adj_machine(const Pos& p) const{
        int ret = 0;
        for(const auto& pp : POSES_EDGE[p.idx()]){
            ret += is_machine(pp);
        }
        return ret;
    }

    bool can_action(const Action& action) const{
        const auto adj_count = [&](const Pos& p){
            int count = 0;
            for(const auto& pp : POSES_EDGE[p.idx()]){
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
            return true;
        }else if(action.kind == MOVE){
            if(!is_machine(action.from)){
                return false;
            }
            if(is_machine(action.to)){
                return false;
            }
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

    void dfs(const Pos& p, uint16_t& count, int& sum_val, float& sum_reserve_val, const bool is_root){
        assert(p.in_range());
        ord[p.idx()] = count;
        count++;
        int root_count = 0;
        if(is_veg(p)){
            sum_val += TP2V[t][p.idx()];
            is_vegs[p.idx()] = false;
        }

        sum_reserve_val += TP2eval[t][p.idx()];
        for(const auto& pp : POSES_EDGE[p.idx()]){
            if(!is_machine(pp)) continue;
            if(checked[pp.idx()] == check_num){
                //後退辺
                chmin(low[p.idx()], ord[pp.idx()]);
                continue;
            }
            root_count++;
            checked[pp.idx()] = check_num;
            dfs(pp, count, sum_val, sum_reserve_val, false);
            chmin(low[p.idx()], low[pp.idx()]);
            if(!is_root && ord[p.idx()] <= low[pp.idx()]){
                is_kansetsu_[p.idx()] = true;
            }
        }

        //Todo:根の関節点判定にバグないかチェック
        if(is_root && root_count >= 2){
            is_kansetsu_[p.idx()] = true;
        }
    }

    void do_turn_end(const vector<Pos>& machines){
        check_num += 2;
        reserve_money = 0;
        max_connect_count = 0;

        is_kansetsu_ = 0;

        for(const auto& base_p : machines){
            if(checked[base_p.idx()] == check_num) continue;
            checked[base_p.idx()] = check_num;
            uint16_t count = ord_root;
            int sum_val = 0;
            float sum_reserve_val = 0;
            dfs(base_p, count, sum_val, sum_reserve_val, true);
            count -= ord_root;
            money += count * sum_val;
            //Todo:center_countをかけるタイミングをちゃんと
            reserve_money += count * sum_reserve_val;
            chmax(max_connect_count, count);

            if(ord_root < N*N*2){
                fill(all(low), 0xffff);
                ord_root = 0xffff;
            }
            ord_root -= N*N*2;

            if((int)count == this->count()) break;
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

    float evaluate() const{
        float eval = 0;
        eval += min(count(), MAX_BUY_COUNT) * (1e9 / MAX_BUY_COUNT);
        eval += money;
        eval += reserve_money;
        return eval;
    }

    bitset<N*N> hash() const{
        return is_vegs ^ is_machines;
    }
};

template<typename Eval, class CenterJudger>
struct BeamSearcher{
    using State = State_tmp<CenterJudger>;
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
            vector<float> dp(N*N);
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
            vector<float> vec_max_val;
            vector<vector<Pos>> vec_max_keiro;
            rep(_t, HOHABA){
                //Todo:あってる？
                const int t = before_state.turn() + _t;
                if(t >= T) break;
                float max_val = -1;
                Pos max_pos = {-1,-1};
                vector<float> dp2(N*N,-INF);
                for(const auto& p : POSES_ALL){
                    if(before_state.is_machine(p)) continue;
                    //Todo:取得済みかどうかのチェック
                    //Todo:累積のほうが良いかも？
                    const float val = [&](){
                        //降ってきてるはずなのに存在しない → 取得済み
                        float ret = 0;
                        bool exist = TP2S[t][p.idx()] > before_state.turn() || before_state.is_veg(p);
                        if(t < START_SAKIYOMI){
                            if(!exist) return 0.0f;
                            //connectしていない場合は何歩目かによって価値が変わる
                            ret += TP2V[t][p.idx()] * (must_connect ? 1 : _t + 1);
                        }else{
                            //Todo:先読みターン数
                            //Todo:提出時にはassert外すかNDEBUG
                            //Todo:must_connectではないときも先読みしたい 3が降ってくる前において、その後隣に置くことで3*2点をしたい
                            assert(must_connect);
                            ret += TP2eval[t][p.idx()];
                            if(exist){
                                ret += TP2V[t][p.idx()];
                            }
                        }
                        return ret;
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

                if(max_val == -1.0f) continue;

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

            //消すものを選ぶ
            //Todo:複数通り
            rep(i, vec_max_keiro.size()){
                const auto& keiro = vec_max_keiro[i];

                vector<Action> actions;
                State after_state = before_state;
                for(const auto& to : keiro){
                    Action action;
                    //Todo:複数回購入できる可能性
                    if(after_state.count() <= MAX_BUY_COUNT && after_state.can_buy()){
                        action.kind = BUY;
                        action.to = to;
                    }else{
                        const auto evaluate = [&](const Pos& from){
                            if(from.manhattan(to) == 1) return -(float)INF;
                            const int t = after_state.turn();
                            return -(TP2eval[t][from.idx()] + after_state.get_veg_value(from));
                        };
                        Pos best_from = {-1,-1};
                        float best_eval = -INF;
                        //must_connectでないときでも、before_machinesの中から関節点ではないものを1つずつ消していくのでこれでよい
                        //Todo:序盤は関節点を削除したほうが評価値が向上する可能性
                        for(const auto& from : before_machines){
                            if(!after_state.is_machine(from)) continue;
                            if(after_state.is_kansetsu(from)) continue;
                            const float eval = evaluate(from);
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
        if(before_state.count() <= NOMUST_CONNECT_THRESHOLD){
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
    // rep(t,T+1){
    //     fill(all(TP2NS[t]),INF);
    // }
    rep(i,M){
        int r,c,s,e,v;cin>>r>>c>>s>>e>>v;
        e++;
        V.push_back({r,c,s,e,v});
        for(int t = s; t < e; ++t){
            TP2V[t][idx(r,c)] += v;
            TP2S[t][idx(r,c)] = s;
            T2P[t].push_back({r,c});
            // TP2eval[t][idx(r,c)] += v;
        }
        constexpr float GAMMA = 0.75;
        float val = v;
        for(int t = s-1; t >= 0; --t){
            val *= GAMMA;
            TP2eval[t][idx(r,c)] += val;
        }
        // if(s > 0){
        //     TP2NS[s-1][idx(r,c)] = s;
        // }
        // TP2V_ruiseki[s][idx(r,c)] += v;

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
    // rep(idx,N*N){
    //     int s = INF;
    //     REP(t,T+1){
    //         chmin(TP2NS[t][idx], s);
    //         s = TP2NS[t][idx];
    //     }
    // }
    // rep(idx, N*N){
    //     rep(t,T){
    //         TP2V_ruiseki[t+1][idx] += TP2V_ruiseki[t][idx];
    //     }
    // }
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
            const Pos&& p = {y,x};
            for(const Dir& dir : DIRS4){
                const Pos&& pp = p + dir;
                if(!pp.in_range()) continue;
                POSES_EDGE[p.idx()].emplace_back(pp);
            }
        }
    }
    POSES_ALL.shrink_to_fit();
    POSES_EDGE.shrink_to_fit();
}

template<class CenterJudger>
pair<vector<Action>, int> solve(){
    State_tmp<CenterJudger> first_state_;
    first_state_.money = 1;
    BeamSearcher<float, CenterJudger> bs_er(first_state_);
    auto&& ans = bs_er.solve();
    cerr<<timer.ms()<<"[ms]"<<endl;
    const int final_money = debug_final_money;
    cerr<<"final money:"<<final_money<<endl;
    //Todo: このreturnは不要なら消す
    return std::make_pair(std::forward<decltype(ans)>(ans), final_money);
}

struct Y14{

};

struct X14{
    static bool is_center(const Pos& p){
        return p.x == N/4 || p.x == N*3/4;
    }
};

struct Y12{
    static bool is_center(const Pos& p){
        return p.y == N/3 || p.y == N*2/3;
    }
};

struct X12{
    static bool is_center(const Pos& p){
        return p.x == N/3 || p.x == N*2/3;
    }
};

int main(int argc, char *argv[]){
    fast_io;

    if(argc >= 2){
        NOMUST_CONNECT_THRESHOLD = stoi(argv[9]);
        START_SAKIYOMI = stoi(argv[11]);
    }

    input();

    // const auto pa1 = solve<Y14>();
    // const auto pa2 = solve<X14>();
    // const auto pa3 = solve<Y12>();
    // const auto pa4 = solve<X12>();
    // constexpr int num = 4;
    // pair<vector<Action>, int> pairs[num] = {pa1,pa2,pa3,pa4};

    // int best_i = 0;
    // int best_score = 0;
    // rep(i,num){
    //     const auto& pa = pairs[i];
    //     const int score = pa.second;
    //     if(score > best_score){
    //         best_i = i;
    //         best_score = score;
    //     }
    // }
    // cout<<pairs[best_i].first<<endl;
    // cerr<<"score:"<<best_score*50<<endl;

    const auto& pa = solve<Y14>();
    cout<<pa.first<<endl;
    cerr<<"score:"<<pa.second*50<<endl;
    return 0;
}
