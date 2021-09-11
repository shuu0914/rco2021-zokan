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
typedef uint16_t HASH_TYPE;

int MAX_BUY_COUNT = 50;
int NOMUST_CONNECT_THRESHOLD = 3;
int START_SAKIYOMI = 100;
constexpr int HASH_STRIDE = 4;
constexpr int HASH_POS_NUM = 8;
constexpr int END_HASH_AREA = 1000;
constexpr float GAMMA_START = 0.0;
constexpr float GAMMA_END = 1.0;
constexpr float SUMI_WEIGHT = 0.8;

const int MAX_HOHABA = 6;
const int BW = 17;

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
    static array<array<int,N*N>,N*N> manhattan_vec;
    int idx_;
    Pos(){}

    Pos(int idx)
    : idx_(idx){}

    Pos(int inp_y, int inp_x)
    : idx_(inp_y*N + inp_x){}

    int manhattan(const Pos& p) const{
        return manhattan_vec[idx()][p.idx()];
    }

    bool operator==(const Pos& p) const{
        return idx() == p.idx();
    }

    bool operator!=(const Pos& p) const{
        return idx() != p.idx();
    }

    //mapに突っ込めるようにするために定義
    bool operator<(const Pos& p) const{
        return idx() < p.idx();
    }

    friend ostream& operator<<(ostream& os, const Pos& p){
        os << "(" << p.idx()/N << "," << p.idx()%N << ")";
        return os;
    }

    string to_answer() const{
        return std::to_string(idx()/N) + " " + std::to_string(idx()%N);
    }

    inline int idx() const{
        return idx_;
    }
};

array<array<int,N*N>,N*N> Pos::manhattan_vec;

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

array<array<int, N*N>, T> TP2V;
array<array<int, N*N>, T> TP2S;
array<array<float, N*N>, T> TP2eval;
vector<vector<Pos>> T2P(T);
vector<vector<Event>> events(T+1);

int debug_final_money = 0;

vector<uint16_t> checked(N*N,0), checked2(N*N,0);
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
vector<vector<vector<Pos>>> POSES_EDGE_DIR(4, vector<vector<Pos>>(N*N));
vector<Pos> POSES_HASH;

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
        // assert(p.in_range());
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
            // assert(p.in_range());
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

    HASH_TYPE hash() const{
        assert(N%HASH_STRIDE == 0);
        HASH_TYPE ret = 0;
        if(t < END_HASH_AREA){
            for(int y_s = 0; y_s < N; y_s += HASH_STRIDE){
                for(int x_s = 0; x_s < N; x_s += HASH_STRIDE){
                    ret *= 2;
                    bool exist = false;
                    for(int y = y_s; y < y_s + HASH_STRIDE; ++y){
                        for(int x = x_s; x < x_s + HASH_STRIDE; ++x){
                            if(is_machines[idx(y,x)]){
                                exist = true;
                                break;
                            }
                        }
                        if(exist) break;
                    }
                    if(exist){
                        ret += 1;
                    }
                }
            }
        }else{
            for(const auto& p : POSES_HASH){
                ret *= 2;
                ret += is_machine(p.idx());
            }
        }
        return ret;
    }
};

array<array<Pos, N*N>, MAX_HOHABA> before_pos;
vector<float> dp(N*N), dp2(N*N);

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
            Pos from = {N*N};
            int max_val = 0;
            Pos to = {N*N};
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
            assert(to.idx() != N*N);

            Action action;
            if(from.idx() == N*N){
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

        const auto func = [&](const int UDLR, const bool must_connect = true){
            if(must_connect){
                fill(all(checked), check_num - 2);
                for(const auto& p : before_machines){
                    checked[p.idx()] = check_num;
                    dp[p.idx()] = 0;
                }
            }else{
                fill(all(checked), check_num);
                fill(all(dp), 0);
            }
            //Todo:正しい？
            const int HOHABA = MAX_HOHABA;
            vector<float> vec_max_val;
            vec_max_val.reserve(HOHABA);
            vector<vector<Pos>> vec_max_keiro;
            vec_max_keiro.reserve(HOHABA);
            rep(_t, HOHABA){
                const auto before_check_num = check_num;
                check_num += 2;
                //Todo:あってる？
                const int t = before_state.turn() + _t;
                if(t >= T) break;
                float max_val = -1;
                Pos max_pos = {-1,-1};
                for(const auto& p : POSES_ALL){
                    if(checked[p.idx()] != before_check_num) continue;
                    for(const Pos& pp : POSES_EDGE_DIR[UDLR][p.idx()]){
                        if(before_state.is_machine(pp)) continue;
                        //Todo:取得済みかどうかのチェック
                        //Todo:累積のほうが良いかも？
                        const float val_add = [&](){
                            //降ってきてるはずなのに存在しない → 取得済み
                            float ret = 0;
                            bool exist = TP2S[t][pp.idx()] > before_state.turn() || before_state.is_veg(pp);
                            if(!must_connect || before_state.turn() < START_SAKIYOMI){
                                if(!exist) return 0.0f;
                                //connectしていない場合は何歩目かによって価値が変わる
                                ret += TP2V[t][pp.idx()] * (must_connect ? 1 : _t + 1);
                            }else{
                                //Todo:先読みターン数
                                //Todo:提出時にはassert外すかNDEBUG
                                //Todo:must_connectではないときも先読みしたい 3が降ってくる前において、その後隣に置くことで3*2点をしたい
                                assert(must_connect);
                                ret += TP2eval[t][pp.idx()];
                                if(exist){
                                    ret += TP2V[t][pp.idx()];
                                }
                            }
                            return ret;
                        }();
                        const float val = val_add + dp[p.idx()];
                        if(checked2[pp.idx()] != before_check_num || dp2[pp.idx()] < val){
                            dp2[pp.idx()] = val;
                            checked2[pp.idx()] = check_num;
                            before_pos[_t][pp.idx()] = p;
                            if(val > max_val && val_add > 0){
                                max_val = val;
                                max_pos = pp;
                            }
                        }
                    }
                }

                swap(dp, dp2);
                swap(checked, checked2);

                if(max_val == -1.0f) continue;

                if(!must_connect && _t+1 < before_state.count()) continue;

                vec_max_val.emplace_back(max_val);
                Pos p = max_pos;
                assert(p.y != -1);
                vector<Pos> kei;
                kei.reserve(_t+1);
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
                actions.reserve(keiro.size());
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
                        Pos best_from = {N*N};
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
                        if(best_from.idx() == N*N){
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

        rep(i,4){
            func(i);
        }
        if(before_state.count() <= NOMUST_CONNECT_THRESHOLD){
            rep(i,4){
                func(i,false);
            }
        }
    }

    vector<Action> back_prop(const int last_idx, int& last_buy_log_idx){
        debug_final_money = logs[last_idx].state.get_money();
        vector<Action> ans;
        ans.reserve(T);
        int idx = last_idx;
        while(idx != 0){
            if(logs[idx].actions.size() == 0){
                ans.emplace_back(logs[idx].action);
                if(last_buy_log_idx == 0 && logs[idx].action.kind == BUY){
                    last_buy_log_idx = idx;
                }
            }else{
                REP(j,logs[idx].actions.size()){
                    const auto& action = logs[idx].actions[j];
                    if(last_buy_log_idx == 0 && logs[idx].actions[j].kind == BUY){
                        last_buy_log_idx = idx;
                    }
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
            unordered_set<HASH_TYPE> S;
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

        int best_score = -INF;
        vector<Action> best_ans;
        int last_buy_log_idx = 0;
        {
            const auto& final_pq = vec_pq[T];
            const auto itr = max_element(all(final_pq),[&](const auto& l, const auto& r){
                return logs[l.second].state.get_money() < logs[r.second].state.get_money();
            });

            //最後のBuyに紐付いたlogのidx
            //そのlogのStateは購入直後の状態
            const int score = logs[itr->second].state.get_money();
            vector<Action> ans = back_prop(itr->second, last_buy_log_idx);
            if(score > best_score){
                best_score = score;
                best_ans = std::move(ans);
            }
        }
        // cerr<<best_score<<endl;

        // //MAX_BUY_COUNTを変えてもう一度
        // MAX_BUY_COUNT += 1;
        // const State& last_buy_state = logs[last_buy_log_idx].state;
        // for(int t = last_buy_state.turn(); t <= T; ++t){
        //     vec_pq[t].clear();
        // }
        // vec_pq[last_buy_state.turn()].emplace_back(logs[last_buy_log_idx].eval, last_buy_log_idx);
        // for(int t = last_buy_state.turn(); t < T; ++t){
        //     auto& current_pq = vec_pq[t];
        //     // partial_sort(current_pq.begin(), current_pq.begin() + min(BW * 2, (int)current_pq.size()), current_pq.end(), greater<>());
        //     sort(all(current_pq), greater<>());
        //     int vec_idx = 0;
        //     unordered_set<HASH_TYPE> S;
        //     for(int _t = 0; _t < BW && vec_idx < current_pq.size(); ++_t, ++vec_idx){
        //         const int idx = current_pq[vec_idx].second;
        //         const auto& state = logs[idx].state;
        //         const auto& hash = state.hash();
        //         if(S.count(hash) > 0){
        //             _t--;
        //             continue;
        //         }
        //         S.insert(hash);

        //         expand(state, idx);
        //     }
        // }
        // {
        //     const auto& final_pq = vec_pq[T];
        //     const auto itr = max_element(all(final_pq),[&](const auto& l, const auto& r){
        //         return logs[l.second].state.get_money() < logs[r.second].state.get_money();
        //     });
        //     //最後のBuyに紐付いたlogのidx
        //     //そのlogのStateは購入直後の状態
        //     const int score = logs[itr->second].state.get_money();
        //     cerr<<score<<endl;
        //     vector<Action> ans = back_prop(itr->second, last_buy_log_idx);
        //     if(score > best_score){
        //         best_score = score;
        //         best_ans = std::move(ans);
        //     }
        // }
        return best_ans;
    }
};

void input(){
    int _; cin>>_>>_>>_;
    timer.start();
    rep(i,TP2V.size()){
        rep(j,TP2V[i].size()){
            TP2V[i][j] = 0;
        }
    }
    rep(i,TP2S.size()){
        rep(j,TP2S[i].size()){
            TP2S[i][j] = 0;
        }
    }
    rep(i,TP2eval.size()){
        rep(j,TP2eval[i].size()){
            TP2V[i][j] = 0.0f;
        }
    }
    rep(i,before_pos.size()){
        rep(j,before_pos[i].size()){
            before_pos[i][j] = {0};
        }
    }
    rep(y,N){
        rep(x,N){
            POSES_ALL.push_back({y,x});
            const Pos&& p = {y,x};
            for(int dy = -1; dy <= 1; dy++){
                for(int dx = -1; dx <= 1; dx++){
                    if(abs(dy)+abs(dx) != 1) continue;
                    if(y+dy>=0 && y+dy<N && x+dx>=0 && x+dx<N){
                        const Pos&& pp = {y+dy, x+dx};
                        POSES_EDGE[p.idx()].emplace_back(pp);
                        if(dy != 0){
                            POSES_EDGE_DIR[((dy+1)>>1)&1][p.idx()].emplace_back(pp);
                            POSES_EDGE_DIR[(((dy+1)>>1)&1)+2][p.idx()].emplace_back(pp);
                        }else{
                            POSES_EDGE_DIR[dx+1][p.idx()].emplace_back(pp);
                            POSES_EDGE_DIR[dx+1+1][p.idx()].emplace_back(pp);
                        }
                    }
                }
            }
        }
    }
    POSES_ALL.shrink_to_fit();
    POSES_EDGE.shrink_to_fit();
    POSES_EDGE_DIR.shrink_to_fit();
    // rep(t,T+1){
    //     fill(all(TP2NS[t]),INF);
    // }
    vector<float> gammas(T);
    rep(t,T){
        const float gamma = GAMMA_START + (GAMMA_END - GAMMA_START) * t / T;
        gammas[t] = gamma;
    }
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
        if(s-1 >= 0){
            const bool is_sumi = r==0 || r==N-1 || c==0 || c==N-1;
            TP2eval[s-1][idx(r,c)] += v * gammas[s-1] * (is_sumi ? SUMI_WEIGHT : 1.0f);
        }
        // constexpr float ALPHA = 0;
        // if(e-2 >= 0){
        //     const Pos base_p = {r,c};
        //     for(const auto& pp : POSES_EDGE[base_p.idx()]){
        //         TP2eval[e-2][pp.idx()] += v * ALPHA;
        //     }
        // }
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

    partial_sort(V.begin(), V.begin() + HASH_POS_NUM, V.end(), [](const Veg& l, const Veg& r){
        return l.v > r.v;
    });
    POSES_HASH.reserve(HASH_POS_NUM);
    rep(i,HASH_POS_NUM){
        POSES_HASH.emplace_back(V[i].r, V[i].c);
    }

    for(const auto& p : POSES_ALL){
        REP(t,T-1){
            TP2eval[t][p.idx()] += TP2eval[t+1][p.idx()] * gammas[t];
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

    for(const Pos& p : POSES_ALL){
        for(const Pos& pp : POSES_ALL){
            const int diff = max(p.idx(), pp.idx()) - min(p.idx(), pp.idx());
            Pos::manhattan_vec[p.idx()][pp.idx()] = diff / N + diff % N;
        }
    }
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
