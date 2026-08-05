#include<bits/stdc++.h>
using namespace std;
int main(){


    int n;
    cin>>n;
    vector<int> x;
    while(n--){
        int k;
        cin>>k;
        x.push_back(k);
    }
    priority_queue<int,vector<int>,greater<int>> heapp;
    int j=x.size();
    int i=0;
    while(j--){
        heapp.push(x[i]);
        i++;
    }
    int b=x.size();

    vector<int> ans;
    while(b--){
    ans.push_back(heapp.top());
    heapp.pop();

    }
    for(auto d:ans){
        cout<<d<<" ";
    }




    return 0;
}