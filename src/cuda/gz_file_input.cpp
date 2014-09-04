#include "file_input.h"
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib> //for exit, atoi etc
#include <cstdio>

using namespace std;

GzFileInput::GzFileInput(const char *szFile)
{
  m_szFileName = szFile;
  m_bHeader = 0;
  m_nHeadLen = 0;
  m_strHeader = 0;
  m_nColumns = gzGetColumns();
  m_nFileLength = gzGetRows();
  if (m_nFileLength * m_nColumns < 5000000)
    m_nRows = m_nFileLength;
  else
    m_nRows = 5000000 / m_nColumns;
  m_nArrayLine1 = 0;
  m_strArray = new string[m_nRows*m_nColumns];
  gzFillArray();
}
GzFileInput::GzFileInput(const char *szFile, bool bHeader)
{
  m_szFileName = szFile;
  m_bHeader = bHeader;
  if (bHeader) {
    m_nHeadLen = gzGetHeadLen();
    m_strHeader = new string[m_nHeadLen];
    gzFillHeader();
  }
  else {
    m_nHeadLen = 0;
    m_strHeader = 0;
  }
  m_nColumns = gzGetColumns();
  cout << "Getting row count" << endl;
  m_nFileLength = gzGetRows();
  if (m_nFileLength * m_nColumns < 5000000)
    m_nRows = m_nFileLength - (int)m_bHeader;
  else
    m_nRows = 5000000 / m_nColumns;
  m_nArrayLine1 = 0;
  m_strArray = new string[m_nRows*m_nColumns];
  gzFillArray();
}
GzFileInput::GzFileInput(const char *szFile, bool bHeader, int nRows, int nColumns)
{
  m_szFileName = szFile;
  m_bHeader = bHeader;
  if (bHeader) {
    m_nHeadLen = gzGetHeadLen();
    m_strHeader = new string[m_nHeadLen];
    gzFillHeader();
  }
  else {
    m_nHeadLen = 0;
    m_strHeader = 0;
  }
  m_nColumns = nColumns;
  m_nFileLength = nRows + (int)m_bHeader;
  if (nRows * nColumns < 5000000)
    m_nRows = nRows;
  else
    m_nRows = 5000000 / m_nColumns;
  m_strArray = new string[m_nRows * m_nColumns];
  m_nArrayLine1 = 0;
  gzFillArray();
}

int GzFileInput::gzGetRows()
{
  string strCmd = "zcat ";
  strCmd.append(m_szFileName);
  FILE *inf = popen(strCmd.c_str(), "r");
  if (!inf) {
    cerr << "Could not load, check file name and directory" << endl;
    exit(1);
  }
  int nRows = 0;
  int ch = getc(inf);
  while (ch != EOF) {
      if (ch == '\n')
	nRows++;
      ch = getc(inf);
  }
  fclose(inf);

  return nRows;
}
int GzFileInput::gzGetColumns()
{
  string strCmd = "zcat ";
  strCmd.append(m_szFileName);
  FILE *inf = popen(strCmd.c_str(), "r");
  if (!inf) {
    cerr << "Could not load, check file name and directory" << endl;
    exit(1);
  }
  if (m_bHeader) {
    char szBuf[10000];
    fgets(szBuf, 10000, inf);
  }
  int nCols = 1;
  int ch = getc(inf);
  while (ch != '\n') {
      if (ch == ' ')
	nCols++;
      ch = getc(inf);
  }
  fclose(inf);
  
  return nCols;
}
int GzFileInput::gzGetHeadLen()
{
  if (!m_bHeader)
    return 0;
  else {
    string strCmd = "zcat ";
    strCmd.append(m_szFileName);
    FILE *inf = popen(strCmd.c_str(), "r");
    if (!inf) {
      cerr << "Could not load, check file name and directory" << endl;
      exit(1);
    }
    int nLen = 1;
    int ch = getc(inf);
    while (ch != '\n') {
      if (ch == ' ')
	nLen++;
      ch = getc(inf);
    }
    fclose(inf);
    return nLen;
  }
}

void GzFileInput::gzFillHeader()
{
  string strCmd = "zcat ";
  strCmd.append(m_szFileName);
  FILE *inf = popen(strCmd.c_str(), "r");
  if (!inf) {
    cerr << "Could not load, check file name and directory" << endl;
    exit(1);
  }
  char szBuf[10000];
  fgets(szBuf, 10000, inf);
  string strLine(szBuf);

  int nColId = 0;
  size_t leftSpace = -1;
  size_t rightSpace = strLine.find_first_of(" ");
  m_strHeader[nColId] = strLine.substr(0, rightSpace);
  for (nColId = 1; nColId < m_nHeadLen; nColId++)
    {
      if (rightSpace == string::npos)  
	m_strHeader[nColId] = "0";
      else
	{
	  leftSpace = rightSpace;
	  rightSpace = strLine.find_first_of(" ", leftSpace + 1);
	  m_strHeader[nColId] = strLine.substr(leftSpace + 1, rightSpace - leftSpace - 1);
	}
    }
  fclose(inf);
  
}

void GzFileInput::gzFillArray()
{
  string strCmd = "zcat ";
  strCmd.append(m_szFileName);
  FILE *inf = popen(strCmd.c_str(), "r");
  if (!inf) {
    cerr << "Could not load, check file name and directory" << endl;
    exit(1);
  }

  char szBuf[10000];
  for (int n = 0; n < m_nArrayLine1 + (int)m_bHeader; n++) {
      fgets(szBuf, 10000, inf);
    }
	 
  for (int rid = 0; rid < m_nRows; rid++) {
    if (m_nArrayLine1 + rid + (int)m_bHeader >= m_nFileLength)
      break;
    
    int cid = 0;
    fgets(szBuf, 10000, inf);
    string strLine(szBuf);

    size_t leftSpace = -1;
    size_t rightSpace = strLine.find_first_of(" ");
    m_strArray[rid * m_nColumns + cid] = strLine.substr(0, rightSpace);
    for (cid = 1; cid < m_nColumns; cid++)
      {
	if (rightSpace == string::npos)  
	  m_strArray[rid * m_nColumns +  cid] = "0";
	else
	  {
	    leftSpace = rightSpace;
	    rightSpace = strLine.find_first_of(" ", leftSpace + 1);
	    m_strArray[rid * m_nColumns + cid] = strLine.substr(leftSpace + 1, rightSpace - leftSpace - 1);
	  }
      }
  }
  fclose(inf);
}

void GzFileInput::getHeader(string strHead[])
{
  for (int h = 0; h < m_nHeadLen; h++) {
    strHead[h] = m_strHeader[h];
  }
}
void GzFileInput::getRow(int anRow[], int nRow)
{
  for (int nC = 0; nC < m_nColumns; nC++) {
    anRow[nC] = getInt(nRow, nC);
  }
}
void GzFileInput::getRow(long alRow[], int nRow)
{
  for (int nC = 0; nC < m_nColumns; nC++) {
    alRow[nC] = getLong(nRow, nC);
  }
}
void GzFileInput::getRow(double adRow[], int nRow)
{
  for (int nC = 0; nC < m_nColumns; nC++) {
    adRow[nC] = getFloat(nRow, nC);
  }
}
void GzFileInput::getRow(string astrRow[], int nRow)
{
  for (int nC = 0; nC < m_nColumns; nC++) {
    astrRow[nC] = getString(nRow, nC);
  }
}

void GzFileInput::getColumn(int anColumn[], int nCol)
{
  int nR = 0;
  while (nR < m_nFileLength - (int)m_bHeader)
    {
      //cout << "Get column, row: " << nR << endl;
      anColumn[nR] = getInt(nR, nCol);
      nR++;
    }
}
//get column as long
void GzFileInput::getColumn(long alColumn[], int nCol)
{
  int nR = 0;
  while (nR < m_nFileLength - (int)m_bHeader)
    {
      alColumn[nR] = getLong(nR, nCol);
      nR++;
    }
}
//get column as double, (or float through implicit 
void GzFileInput::getColumn(double adColumn[], int nCol)
{
  int nR = 0;
  while (nR < m_nFileLength - (int)m_bHeader)
    {
      adColumn[nR] = getFloat(nR, nCol);
      nR++;
    }
}
//get column as string
void GzFileInput::getColumn(string astrColumn[], int nCol)
{
  int nR = 0;
  while (nR < m_nFileLength - (int)m_bHeader)
    {
      astrColumn[nR] = getString(nR, nCol);
      nR++;
    }
}

int GzFileInput::getInt(int nRow, int nCol)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    gzFillArray();
    nARow = 0;
    //cout << "Getting row: " << nRow << endl;
  }
  return atoi(m_strArray[nARow * m_nColumns + nCol].c_str());
}
long GzFileInput::getLong(int nRow, int nCol)
{
    int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    gzFillArray();
    nARow = 0;
  }
  return atol(m_strArray[nARow * m_nColumns + nCol].c_str());
}
double GzFileInput::getFloat(int nRow, int nCol)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    gzFillArray();
    nARow = 0;
    //cout << "Getting row: " << nRow << endl;
  }
  return atof(m_strArray[nARow * m_nColumns + nCol].c_str());
}
string GzFileInput::getString(int nRow, int nCol)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    gzFillArray();
    nARow = 0;
  }
  return m_strArray[nARow * m_nColumns + nCol];
}

int GzFileInput::getHeadInt(int nCol)
{
  return atoi(m_strHeader[nCol].c_str());
}
long GzFileInput::getHeadLong(int nCol)
{
  return atol(m_strHeader[nCol].c_str());
}
double GzFileInput::getHeadFloat(int nCol)
{
  return atof(m_strHeader[nCol].c_str());
} 
string GzFileInput::getHeadString(int nCol)
{
  return m_strHeader[nCol].c_str();
}

