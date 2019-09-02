#include <bits/stdc++.h>

using namespace std;

#define int long long
#define endl "\n"

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n, x;
	cin >> n;
	vector<int> v, lis;
	for(int i=0; i < n; i++) {
		cin >> x;
		v.push_back(x);
	}
	lis.push_back(v[0]);
	for(int i=1; i < v.size(); i++) {
		auto it = lower_bound(lis.begin(), lis.end(), v[i]);
		if(it == lis.end())
			lis.push_back(v[i]);
		else
			*it = v[i];
	}
	cout << lis.size() << endl;
}
