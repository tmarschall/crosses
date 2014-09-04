/*
 *  file_input.h
 *  
 *
 *  Created by Theodore Marschall on 7/9/10.
 *
 *  Last Modified 12/17/2013
 *    Changes:
 *     -New GzFileInput class for opening gzipped files
 *     -Better header handling & functions in DatFileInput (also in GzFileInput)
 *     -getRow() functions to get rows of same data type at once
 *
 */

#ifndef FILE_INPUT_H
#define FILE_INPUT_H

#include <string>
#include <fstream>

using namespace std;

//dat files are assumed to contain columns of data with 
class DatFileInput
{
private:
	const char* m_szFileName;
	bool m_bHeader;
	int m_nFileLength;
	int m_nColumns;
	int m_nRows;  //the number of rows in the file includes the header row if the file has one
	int m_nHeadLen;
	string* m_strArray;
	string* m_strHeader;
	int m_nArrayLine1;
	
	int datGetRows();
	int datGetColumns();
	int datGetHeadLen();
	void datFillHeader();
	void datFillArray();
public:
	//constructors
	DatFileInput(const char* szFile);
	DatFileInput(const char* szFile, bool bHeader);
	DatFileInput(const char* szFile, int nRows, int nColumns);
	DatFileInput(const char* szFile, bool bHeader, int nRows, int nColumns);
	//destructor
	~DatFileInput() { delete[] m_strArray; delete[] m_strHeader; }
	
	void datDisplayArray();
	
	void getColumn(int anColumn[], int nColumn);
	void getColumn(long alColumn[], int nColumn);
	void getColumn(double adColumn[], int nColumn);
	void getColumn(string astrColumn[], int nColumn);
	void getHeader(string strHead[]);
	void getRow(int anRow[], int nRow);
	void getRow(long alRow[], int nRow);
	void getRow(double adRow[], int nRow);
	void getRow(string astrRow[], int nRow);
	
	int getInt(int nRow, int nColumn);
	long getLong(int nRow, int nColumn);
	double getFloat(int nRow, int nColumn);
	string getString(int nRow, int nColumn);

	int getHeadInt(int nCol);
	long getHeadLong(int nCol);
	double getHeadFloat(int nCol);
	string getHeadString(int nCol);
	
	int getColumns() { return m_nColumns; }
	int getRows() { return m_nFileLength - static_cast<int>(m_bHeader); }
	int getHeadLen() { return m_nHeadLen; }
};

class GzFileInput
{
private:
  const char* m_szFileName;
  bool m_bHeader;
  int m_nFileLength;
  int m_nColumns;
  int m_nRows;
  int m_nHeadLen;
  
  string *m_strArray;
  string *m_strHeader;
  int m_nArrayLine1;
  
  int gzGetRows();
  int gzGetColumns();
  int gzGetHeadLen();
  void gzFillArray();
  void gzFillHeader();
 public:
  GzFileInput(const char *szFile);
  GzFileInput(const char *szFile, bool bHeader);
  GzFileInput(const char *szFile, bool bHeader, int nRows, int nColumns);
  ~GzFileInput() { delete[] m_strArray; delete[] m_strHeader; }

  void getHeader(string strHead[]);
  void getRow(int pnRow[], int nRow);
  void getRow(long plRow[], int nRow);
  void getRow(double pdRow[], int nRow);
  void getRow(string pstrRow[], int nRow);
  void getColumn(int pnColumn[], int nCol);
  void getColumn(long plColumn[], int nCol);
  void getColumn(double pdColumn[], int nCol);
  void getColumn(string pstrColumn[], int nCol);

  int getInt(int nRow, int nCol);
  long getLong(int nRow, int nCol);
  double getFloat(int nRow, int nCol);
  string getString(int nRow, int nCol);

  int getHeadInt(int nCol);
  long getHeadLong(int nCol);
  double getHeadFloat(int nCol);
  string getHeadString(int nCol);

  int getColumns() { return m_nColumns; }
  int getRows() { return m_nFileLength - (int)m_bHeader; }
  int getHeadLen() { return m_nHeadLen; }
};


class BinFileInput
{
private:
  ifstream m_FileStream;
  long int m_nPosition;
  long int m_nLength;
  int m_nCharSize;
  int m_nShortSize;
  int m_nIntSize;
  int m_nFloatSize;
  int m_nLongSize;
  int m_nDoubleSize;
  char *m_szBuffer;

public:
  BinFileInput(const char* szInput);
  ~BinFileInput();

  void set_position(long int nPosition);
  void skip_bytes(long int nBytes);
  void jump_back_bytes(long int nBytes);

  char getChar();
  short getShort();
  int getInt();
  float getFloat();
  long getLong();
  double getDouble();

  long int getFileSize() { return m_nLength; }
};


#endif
