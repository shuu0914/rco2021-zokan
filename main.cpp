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
constexpr int HOHABA = 6;
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
    bitset<N*N> is_kansetsu = 0;
    int reserve_money = 0;
    int max_connect_count = 0;
    int adj_pena = 0;

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

    void dfs(const Pos& p, bitset<N*N>& checked, int& count, int& sum_val, int& sum_reserve_val){
        assert(p.in_range());
        count++;
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
            if(checked[pp.idx()]) continue;
            checked[pp.idx()] = true;
            dfs(pp, checked, count, sum_val, sum_reserve_val);
        }
        if(adj_count >= 3){
            adj_pena++;
        }
    }

    void do_turn_end(const vector<Pos>& machines){
        bitset<N*N> checked = 0;
        reserve_money = 0;
        max_connect_count = 0;
        adj_pena = 0;

        for(const auto& base_p : machines){
            if(checked[base_p.idx()]) continue;
            checked[base_p.idx()] = true;
            int count = 0;
            int sum_val = 0;
            int sum_reserve_val = 0;
            dfs(base_p, checked, count, sum_val, sum_reserve_val);
            money += count * sum_val;
            reserve_money += count * sum_val;
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
        eval -= adj_pena * 10;
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

        //Todo:moveになってる？
        Log(State&& state_, Action&& action_, const int before_idx_, const Eval eval_)
        : state(state_), action(action_), before_idx(before_idx_), eval(eval_)
        {

        }

        Log(State&& state_, const Action& action_, const int before_idx_, const Eval eval_)
        : state(state_), action(action_), before_idx(before_idx_), eval(eval_)
        {

        }

        //Todo:moveになってる？
        Log(State&& state_, vector<Action>&& actions_, const int before_idx_, const Eval eval_)
        : state(state_), actions(actions_), before_idx(before_idx_), eval(eval_)
        {

        }

        Log(State&& state_, const vector<Action>& actions_, const int before_idx_, const Eval eval_)
        : state(state_), actions(actions_), before_idx(before_idx_), eval(eval_)
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
    vector<priority_queue<Log>> vec_pq;

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
            vec_pq[t].emplace(std::move(after_state), action, before_idx, eval);
        };
        const auto push_actions = [&](const vector<Action>& actions, State&& after_state, const int bonus = 0){
            const int t = after_state.turn();
            assert(t <= T);
            const Eval eval = after_state.evaluate() + bonus;
            vec_pq[t].emplace(std::move(after_state), actions, before_idx, eval);
        };

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

        const auto func = [&](const auto UD, const auto LR){
            vector<int> dp(N*N,-INF);
            for(const auto& p : POSES_ALL){
                if(before_state.is_machine(p)){
                    dp[p.idx()] = 0;
                }
            }
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
                            return TP2V[t][p.idx()];
                        }else{
                            //Todo:先読みターン数
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
                const auto& val = vec_max_val[i];
                assert(keiro.size() > 0);
                vector<Pos> vec_bfs;
                vec_bfs.push_back(keiro.front());
                bitset<N*N> checked = 0;
                assert(vec_bfs.size() > 0);
                assert(vec_bfs.front().in_range());
                checked[vec_bfs.front().idx()] = true;
                int idx = 0;
                while(idx != vec_bfs.size()){
                    const Pos p = vec_bfs[idx];
                    idx++;

                    for(const auto& dir : DIRS4){
                        const Pos&& pp = p + dir;
                        if(!pp.in_range()) continue;
                        if(checked[pp.idx()]) continue;
                        if(!before_state.is_machine(pp)) continue;
                        checked[pp.idx()] = true;
                        vec_bfs.push_back(pp);
                    }
                }

                // for(const auto& debug : keiro){
                //     cerr<<debug<<" ";
                // }
                // cerr<<endl;
                // cerr<<"hoge:";
                // for(const auto& p : POSES_ALL){
                //     if(before_state.is_machine(p)){
                //         cerr<<p;
                //     }
                // }
                // cerr<<endl;

                //遠いものから
                //Todo:消すものが足りない場合に追加したものをそのまま消す可能性
                //先頭は一歩目なので無視
                auto itr = vec_bfs.end();
                assert(vec_bfs.size() > 0);
                itr--;
                vector<Action> actions;
                State after_state = before_state;
                for(const auto& to : keiro){
                    Action action;
                    //Todo:複数回購入できる可能性
                    if(after_state.turn() <= MAX_BUY_T && after_state.can_buy()){
                        action.kind = BUY;
                        action.to = to;
                    }else{
                        if(itr == vec_bfs.begin()) break;
                        action.kind = MOVE;
                        action.from = *itr;
                        action.to = to;
                        --itr;
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
        logs.reserve((int)1e7);
        {
            Log first_log;
            first_log.set_state(first_state);
            logs.emplace_back(first_log);
        }
        int logs_start = 0;
        rep(t, T){
            for(int before_idx = logs_start; before_idx < logs.size(); ++before_idx){
                // cerr<<before_idx<<endl;
                //Todo:参照にしたい
                const State before_state = logs[before_idx].state;
                expand(before_state, before_idx);
            }

            priority_queue<Log>& next_pq = vec_pq[t+1];
            logs_start = logs.size();
            unordered_set<bitset<N*N>> S;
            for(int _t = 0; _t < BW && next_pq.size() > 0; ++_t){
                Log log = next_pq.top(); next_pq.pop();
                const auto& hash = log.state.hash();
                if(S.count(hash) > 0){
                    _t--;
                    continue;
                }
                S.insert(hash);
                logs.emplace_back(std::move(log));
            }
        }

        const vector<Action> ans = back_prop(logs_start);
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