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
};

int idx(const int y, const int x){
    return y * N + x;
}

int main(void){
    fast_io;

    return 0;
}