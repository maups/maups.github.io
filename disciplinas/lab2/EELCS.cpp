#include <bits/stdc++.h>

using namespace std;

#define int long long
#define endl "\n"

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	string s, t;
	cin >> s >> t;
	int dp[s.size()+1][t.size()+1];
	for(int i=0; i <= s.size(); i++)
		dp[i][0] = 0;
	for(int j=0; j <= t.size(); j++)
		dp[0][j] = 0;
	for(int i=1; i <= s.size(); i++)
		for(int j=1; j <= t.size(); j++)
			if(s[i-1] == t[j-1])
				dp[i][j] = dp[i-1][j-1]+1;
			else
				dp[i][j] = max(dp[i][j-1], dp[i-1][j]);
	cout << dp[s.size()][t.size()] << endl;
}
