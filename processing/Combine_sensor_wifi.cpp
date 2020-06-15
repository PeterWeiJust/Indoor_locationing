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
double caldistance(double wifix, double wifiy, double sensorx, double sensory){
	return sqrt(pow(wifix - sensorx, 2) + pow(wifiy - sensory, 2));
}

string wifiroute = "Timed Data/Scenario_1/scenario1-route";
string sensorroute = "Scenario_1/";
string outputroute = "combined_data/sensor_wifi_1_";

int main(){
	
	string value;
	ifstream finwifi, finsensor;
	for (int k = 1; k < 9; k++){
		finwifi.open(wifiroute + to_string(k) + ".csv", ios::in);

		vector<vector<string>>wifidata;
		vector<vector<string>>sensordata;
		while (finwifi.good())
		{
			getline(finwifi, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");

			wifidata.push_back(wifirow);
		}

		finwifi.close();
		finwifi.clear();


		finsensor.open(sensorroute + to_string(k) + "_timestep100_unique.csv", ios::in);
		while (finsensor.good())
		{
			getline(finsensor, value);
			vector<string>wifirow;
			SplitString(value, wifirow, ",");
			sensordata.push_back(wifirow);
		}
		finsensor.close();
		finsensor.clear();

		for (int i = 1; i < sensordata.size() - 1; i++){
			double sensortime = atof(sensordata[i][0].c_str());
			double mindistance = 10000000;
			int minindex = -1;
			for (int j = 1; j < wifidata.size() - 1; j++){
				double wifitime = atof(wifidata[j][2].c_str());
				double dis = abs(sensortime - wifitime);
				if (dis < mindistance && wifidata[j][46] == "-1"){
					mindistance = dis;
					minindex = j;
				}
			}
			if (mindistance <= 1000 && minindex != -1 && minindex != 0){
				wifidata[minindex][46] = "1";
				sensordata[i].insert(sensordata[i].end(), wifidata[minindex].begin() + 3, wifidata[minindex].end() - 1);
				sensordata[i].push_back(to_string(mindistance));
			}
			else{
				for (int k = 0; k < 43; k++){
					sensordata[i].push_back("0");
				}
				sensordata[i].push_back("-1");
			}

		}

		ofstream out(outputroute + to_string(k) + "_timestep100.csv", ios::out);
		string columindex = "";
		for (int j = 0; j < 60; j++){

			columindex += to_string(j) + ",";

		}
		out << columindex << endl;
		for (int i = 1; i< sensordata.size(); ++i)
		{
			string rawdata = "";

			for (int j = 0; j < sensordata[i].size(); j++){
				rawdata += sensordata[i][j] + ",";
			}

			out << rawdata << endl;
		}
	}
	

	return 0;
}