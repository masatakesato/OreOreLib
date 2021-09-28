// https://hotnews8.net/programming/tricky-code/c-code02

#include <windows.h>
#include <conio.h>

#include	<oreore/common/TString.h>




#define ESC 27




void func()
{
	int val=0;
	for( int i=0; i<100000; ++i )
	{
		for( int j=0; j<10000; ++j )
			val++;

		for( int j=0; j<10000; ++j )
			val--;
	}
}



int main(void)
{
    int cnt = 0;

    while (1) 
	{
		func();


        if (_kbhit() && _getch() == ESC)
		{
			tcout << "You Hit Esc-Key!!\n";
			//_cputs("\r\nYou Hit Esc-Key!!\r\n");
            break;
        }

        tcout << ++cnt << tendl;
    }

    return 0;
}