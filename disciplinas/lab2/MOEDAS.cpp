#include <bits/stdc++.h>

using namespace std;

#define int long long

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n;
	while(cin >> n, n != 0) {
		int m, valor;
		cin >> m;
		vector<int> moedas;
		for(int i=0; i < m; i++) {
			cin >> valor;
			moedas.push_back(valor);
		}
		int dp[n+1];
		for(int i=0; i <= n; i++)
			dp[i] = -1;
		dp[0] = 0;
		for(int i=0; i < m; i++)
			for(int j=0; j <= n; j++)
				if(dp[j] != -1 && j+moedas[i] <= n)
					if(dp[j+moedas[i]] == -1)
						dp[j+moedas[i]] = dp[j]+1;
					else
						dp[j+moedas[i]] = min(dp[j+moedas[i]], dp[j]+1);
		if(dp[n] == -1)
			cout << "Impossivel" << endl;
		else
			cout << dp[n] << endl;
	}
}
