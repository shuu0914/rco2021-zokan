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
#include <deque>
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
constexpr int N = 16, M = 5000, T = 1000;
typedef uint32_t HASH_TYPE;

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
const int BW = 100;
constexpr int MAX_DEPTH = 6;

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

vector<vector<int>> TP2V(T+1, vector<int>(N*N));
vector<vector<int>> TP2S(T+1, vector<int>(N*N));
vector<vector<int>> TP2NS(T+1, vector<int>(N*N));
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

ostream& operator<<(ostream& os, const vector<Action>& actions){
    rep(i, actions.size()){
        //途中まで出力してデバッグしたい時用
        if(true || i <= 631){
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
        }else{
            os<<-1;
            if(i+1 != actions.size()){
                os<<endl;
            }
        }
    }
    return os;
}

struct State{
    int money = 1;
    int reserve_money = 0;
    int t = 0;
    bitset<N*N> is_machines = 0;
    //最後に存在していたt
    array<int, N*N> last_pass = {};
    //Todo:RingBufferで高速化
    //t→val
    map<int,int> expected_vegs;
    deque<Pos> machines;

    State(){
        fill(all(last_pass), -1);
    }

    int get_cost() const{
        return (machines.size() + 1) * (machines.size() + 1) * (machines.size() + 1);
    }

    bool can_buy() const{
        return get_cost() <= money;
    }

    bool is_machine(const Pos& p) const{
        return is_machines[p.idx()];
    }

    bool is_veg(const Pos& p) const{
        return TP2V[t][p.idx()] > 0 && (TP2S[t][p.idx()] == t || TP2S[t][p.idx()] > last_pass[p.idx()]);
    }

    int get_veg_value(const Pos& p) const{
        if(!is_veg(p)){
            return 0;
        }
        return TP2V[t][p.idx()];
    }

    int count() const{
        return machines.size();
    }

    int turn() const{
        return t;
    }

    int get_money() const{
        return money;
    }

    Pos get_tail() const{
        return machines.front();
    }

    Pos get_head() const{
        return machines.back();
    }

    vector<Pos> get_machines() const{
        vector<Pos> ret(all(machines));
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

    void do_action(const Action& action){
        assert(can_action(action));

        const auto update_money = [&](const Pos& to){
            //即時報酬
            money += get_veg_value(to) * count();
            //将来報酬
            int now = t;
            while(true){
                const int next = TP2NS[now][to.idx()];
                //Todo:戦略的パスやBUYによる時間稼ぎを考慮
                if(next >= t + count()){
                    break;
                }
                const int val = TP2V[next][to.idx()];
                expected_vegs[next] += val;
                reserve_money += val * count();
                now = next;
            }
            //turn() + count()の終了時まで存在する
            last_pass[to.idx()] = turn() + count();
        };

        if(action.kind == BUY){
            money -= get_cost();
            is_machines[action.to.idx()] = true;
            assert(count() == 0 || reserve_money%count() == 0);
            //計算し直し
            if(count() > 0){
                reserve_money = reserve_money / count() * (count()+1);
            }
            rep(i, machines.size()){
                const auto& p = machines[i];
                last_pass[p.idx()]++;
                //turn() + iに降ってくる報酬はもらえない換算だったが貰えるようになった
                const int ttt = turn() + i;
                if(ttt >= T-1) break;
                const int val = TP2V[ttt][p.idx()];
                if(val > 0 && TP2S[ttt][p.idx()] == ttt){
                    reserve_money += val * (count() + 1);
                    expected_vegs[ttt] += val;
                }
            }

            machines.emplace_back(action.to);

            update_money(action.to);
        }else if(action.kind == MOVE){
            is_machines[action.from.idx()] = false;
            is_machines[action.to.idx()] = true;
            assert(machines.front() == action.from);
            machines.pop_front();
            machines.emplace_back(action.to);

            update_money(action.to);
        }else{
            //PASS
        }
        do_turn_end();
    }

    void do_turn_end(){
        const auto itr = expected_vegs.find(t);
        if(itr != expected_vegs.end()){
            const int veg_num = itr->second;
            const int val = veg_num * count();
            money += val;
            reserve_money -= val;
            expected_vegs.erase(itr);
        }
        t++;
    }

    void dfs(vector<pair<vector<Action>, int>>& best_actions, vector<Action>& actions, const int depth, const int eval_sum, const int max_depth){
        if(depth > 0 && best_actions[depth-1].second < eval_sum){
            best_actions[depth-1].second = eval_sum;
            best_actions[depth-1].first = actions;
        }
        if(depth == max_depth){
            return;
        }
        //Todo:簡略化
        //Todo:買えるなら買う
        const Pos head = get_head();
        for(const auto& dir : DIRS4){
            Pos&& to = head + dir;
            if(!to.in_range()) continue;
            if(is_machine(to)) continue;

            //turn() + count - 1の終了時まで存在
            const auto calc_eval = [&](const Pos& to){
                int eval = 0;
                //即時報酬
                eval += get_veg_value(to);
                //将来報酬
                int now = turn();
                while(true){
                    const int next = TP2NS[now][to.idx()];
                    //Todo:戦略的パスやBUYによる時間稼ぎを考慮
                    if(next >= turn()){
                        break;
                    }
                    const int val = TP2V[next][to.idx()];
                    eval += val;
                    now = next;
                }
                return eval;
            };
            const int eval = calc_eval(to);

            Action action;
            action.kind = MOVE;
            action.from = get_tail();
            action.to = to;
            actions.emplace_back(action);

            machines.pop_front();
            machines.emplace_back(action.to);

            t++;

            dfs(best_actions, actions, depth+1, eval_sum+eval, max_depth);

            t--;

            machines.pop_back();
            machines.emplace_front(action.from);

            actions.pop_back();
        }
    }

    int evaluate() const{
        int eval = 0;
        eval += count() * (t < MAX_BUY_T ? 1e9 / (N*N) : 0);
        eval += money + reserve_money;
        return eval;
    }

    HASH_TYPE hash() const{
        return get_head().idx();
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
        const auto push_actions = [&](vector<Action>&& actions, const int bonus = 0){
            State after_state = before_state;
            for(auto& action : actions){
                //Todo:vector<Pos> actionsを渡す形で実装し直す
                if(action.kind == MOVE && after_state.can_buy()){
                    action.kind = BUY;
                }
                //Todo:これはその場しのぎ
                action.from = after_state.get_tail();
                //Todo:BUYに変わったことによりタイミングがずれて動けなくなることへの対処
                if(!after_state.can_action(action)){
                    return;
                }
                after_state.do_action(action);
            }
            const int t = after_state.turn();
            assert(t <= T);
            const Eval eval = after_state.evaluate() + bonus;
            vec_pq[t].emplace_back(eval, logs.size());
            logs.emplace_back(std::forward<State>(after_state), std::forward<vector<Action>>(actions), before_idx, eval);
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

        const int depth = min(MAX_DEPTH, T - before_state.turn());
        vector<pair<vector<Action>, int>> results(depth);
        State state = before_state;
        {
            vector<Action> actions;
            state.dfs(results, actions, 0, 0, depth);
        }
        rep(i,results.size()){
            auto&& actions = results[i].first;
            if(actions.size() == 0) continue;
            push_actions(std::move(actions));
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
            //Todo:調整し直し
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
            // cerr<<t<<" "<<current_pq.size()<<endl;
            // partial_sort(current_pq.begin(), current_pq.begin() + min(BW * 2, (int)current_pq.size()), current_pq.end(), greater<>());
            sort(all(current_pq), greater<>());
            int vec_idx = 0;
            unordered_set<HASH_TYPE> S;
            for(int _t = 0; _t < BW && vec_idx < current_pq.size(); ++_t, ++vec_idx){
                const int idx = current_pq[vec_idx].second;
                const auto& state = logs[idx].state;
                if(t > 0){
                    const auto& hash = state.hash();
                    if(S.count(hash) > 0){
                        _t--;
                        continue;
                    }
                    S.insert(hash);
                }
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

pair<vector<Action>, int> solve(){
    State first_state_;
    first_state_.money = 1;
    BeamSearcher<int> bs_er(first_state_);
    auto&& ans = bs_er.solve();
    cerr<<timer.ms()<<"[ms]"<<endl;
    const int final_money = debug_final_money;
    cerr<<"final money:"<<final_money<<endl;
    return std::make_pair(std::forward<decltype(ans)>(ans), final_money);
}

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
    const auto& pa = solve();
    cout<<pa.first<<endl;
    cerr<<"score:"<<pa.second*50<<endl;
    return 0;
}
