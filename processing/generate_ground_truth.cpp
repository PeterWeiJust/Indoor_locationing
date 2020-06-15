// RomaniaData.cpp : 定义控制台应用程序的入口点。
//

#include"stdafx.h"
#include<stdio.h>
#include<windows.h>
#include<fstream>        
#include<sstream>        
#include<list>
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<cstdlib>
using namespace std;
void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}
string wifiroute = "converteddata/Scenario_1_";
string sensorroute = "Scenario_1/";

int main()
{

	
	for (int k = 1; k <= 8; k++){
		string value;
		ifstream finwifi, finsensor;
		finwifi.open(wifiroute + to_string(k) + "_converted.csv", ios::in);

		vector<vector<string>>sensordata;
		vector<vector<string>>sensorlabel;
		vector<vector<string>>res;
		while (finwifi.good())
		{
			getline(finwifi, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");

			sensordata.push_back(wifirow);
		}

		finwifi.close();
		finwifi.clear();


		finsensor.open(sensorroute + to_string(k) + "/ground_truth_1.csv", ios::in);
		while (finsensor.good())
		{
			getline(finsensor, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");
			sensorlabel.push_back(wifirow);
		}
		finsensor.close();
		finsensor.clear();


		for (int i = 1; i < sensorlabel.size() - 1; i++){
			string time = sensorlabel[i][0];
			vector<string>subtime;
			SplitString(time, subtime, ":");
			for (int j = 1; j < sensordata.size() - 1; j++){
				if (sensordata[j][2] != "nan"){
					string r2 = sensordata[j][2];
					vector<string>subr2, timesubr2;
					SplitString(r2, subr2, " ");
					SplitString(subr2[1], timesubr2, ":");
					timesubr2[3] += "0";
					if (subtime[3].length() > 3){
						subtime[3] = subtime[3].substr(0, 3);
					}
					if (timesubr2[0] == subtime[0] && timesubr2[1] == subtime[1] && timesubr2[2] == subtime[2] && timesubr2[3] >= subtime[3]){
						int times = stoi(sensordata[j][1]);
						int g = times + (100 - times % 100);
						sensorlabel[i][0] = to_string(g);
						res.push_back(sensorlabel[i]);
						break;
					}
				}
			}
		}

		ofstream out(sensorroute + to_string(k) + "/"+"1.csv", ios::out);
		string columindex = "";
		columindex = "Time,lat1,Lng1,x,y,lat,Lng";

		out << columindex << endl;
		for (int i = 0; i< res.size(); ++i)
		{
			string rawdata = "";

			for (int j = 0; j < res[i].size(); j++){
				rawdata += res[i][j] + ",";
			}

			out << rawdata << endl;
		}
	}
	
	return 0;
}