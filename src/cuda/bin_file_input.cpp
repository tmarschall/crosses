/*
 *  bin_file_input.cpp
 *  
 *
 *  Created by Theodore Marschall on 3/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "file_input.h"
#include <fstream>
#include <assert.h>
#include <cstdlib> //for exit, atoi etc

using namespace std;


BinFileInput::BinFileInput(const char* szFile)
{ 
  m_FileStream.open(szFile, ios::in | ios::binary);

  m_nPosition = 0;
  m_FileStream.seekg(0, ios::end);
  m_nLength = m_FileStream.tellg();
  m_FileStream.seekg(0, ios::beg);

  m_nCharSize = sizeof(char);
  m_nShortSize = sizeof(short);
  m_nIntSize = sizeof(int);
  m_nFloatSize = sizeof(float);
  m_nLongSize = sizeof(long);
  m_nDoubleSize = sizeof(double);

  m_szBuffer = 0;
}

BinFileInput::~BinFileInput()
{
  delete[] m_szBuffer;
  m_FileStream.close();
}

void BinFileInput::set_position(long int nPosition)
{
  assert(nPosition < m_nLength);

  m_FileStream.seekg(nPosition);
  m_nPosition = nPosition;
}

void BinFileInput::skip_bytes(long int nBytes)
{
  m_nPosition = m_FileStream.tellg();
  m_nPosition += nBytes;
  assert(m_nPosition <= m_nLength);

  m_FileStream.seekg(m_nPosition);

}

void BinFileInput::jump_back_bytes(long int nBytes)
{
  m_nPosition = m_FileStream.tellg();
  m_nPosition -= nBytes;
  assert(m_nPosition > 0);

  m_FileStream.seekg(m_nPosition);
}

char BinFileInput::getChar()
{
  char *cChar;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nCharSize];
  m_FileStream.read(m_szBuffer, m_nCharSize);
  cChar = reinterpret_cast<char*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *cChar;
}

short BinFileInput::getShort()
{
  short *snShort;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nShortSize];
  m_FileStream.read(m_szBuffer, m_nShortSize);
  snShort = reinterpret_cast<short*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *snShort;
}

int BinFileInput::getInt()
{
  int *nInt;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nIntSize];
  m_FileStream.read(m_szBuffer, m_nIntSize);
  nInt = reinterpret_cast<int*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *nInt;
}

long BinFileInput::getLong()
{
  long *lnLong;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nLongSize];
  m_FileStream.read(m_szBuffer, m_nLongSize);
  lnLong = reinterpret_cast<long*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *lnLong;
}

float BinFileInput::getFloat()
{
  float *fFloat;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nFloatSize];
  m_FileStream.read(m_szBuffer, m_nFloatSize);
  fFloat = reinterpret_cast<float*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *fFloat;
}

double BinFileInput::getDouble()
{
  double *dDouble;
  delete[] m_szBuffer;
  m_szBuffer = new char[m_nDoubleSize];
  m_FileStream.read(m_szBuffer, m_nDoubleSize);
  dDouble = reinterpret_cast<double*>(m_szBuffer);
  m_nPosition = m_FileStream.tellg();

  return *dDouble;
}
