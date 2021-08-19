// ランレングス圧縮テストプログラム



#include <stdio.h>


#define Elements 18// 要素数
#define MaxRleLen 18// ランレングス圧縮最大個数
#define MaxUnLen 18// 非圧縮最大個数



//-------------- data[]のfirstからiter個連続して書き込む --------------//
void Write_uncompressed(unsigned int first, unsigned int iter, unsigned char *data){
	
	if(iter==0) return;

	for(unsigned int k=0; k<iter; k+=MaxUnLen){ 
		
		if((iter-k)>MaxUnLen){ 
			printf("非圧縮データ：%d個\n", MaxUnLen);
			for(unsigned int l=0; l<MaxUnLen; l++) printf("   %d\n", data[k+l+first]);
		}
		else{ 
			printf("非圧縮データ：%d個\n", iter-k);
			for(unsigned int l=0; l<iter-k; l++) printf("   %d\n", data[k+l+first]);
		}

		printf("\n");

	}// end of k loop

}


//---------------------- dataをiter個ランレングス圧縮で書き込む ------------------------//
void Write_RLE(unsigned char data, unsigned int iter){
	
	if(iter==0) return;
	
	// iter個が上限オーバーしてるときは分割
	for(unsigned int k=0; k<iter; k+=MaxRleLen){ 

			printf("RLEデータ2：\n");
			printf("データ：%d,  反復回数：%d\n\n", data, iter-k);

	}// end of k loop


}






int main(int argc, char *argv[]){
	
	unsigned char data[Elements] = {0,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1};//{1,1,1,1, 2,2,2, 1,1, 2,1,2,1,2, 1,1, 1,1};// RLE対象データ
	unsigned int pivot = 0;
	unsigned int j =0;
	
	
	printf("元データ： \n");
	for (unsigned int k=0; k<18; k++) printf("%d\n", data[k]);
	printf("\n");
	
	
	///////////////////////// データのランレングス圧縮 /////////////////////////
	
	while(1){
		
		if(j>=Elements){
			Write_uncompressed(pivot,Elements-pivot,data);
			break;
		}

		if(data[j]==data[j+1]){// RLEエリアに入った
			
			//--------- pivotからjまでを非圧縮で書き込み ----------//
			Write_uncompressed(pivot,j-pivot,data);
			
			//--------- 圧縮できるデータの個数を数える ----------//
			unsigned int l;// 自分の1つ後ろから調べる
			
			for(l=2; l<Elements-j; l++){// 圧縮最大連続数を超えたら切る
				if(data[j]!=data[j+l] || l == MaxRleLen) break;
			}
	
			//--------- data[j]をl個RLEで書き込み ----------//
			Write_RLE(data[j],l);
			
			j += l;
			pivot = j;
			
		}
		else{ j++; }
		
	}// end of while
	
return 0;
}