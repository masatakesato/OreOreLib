#ifndef SOCKET_H
#define	SOCKET_H


#include	<exception>
//https://setuna-kanata.hatenadiary.org/entry/20100131/1264946168

class SocketException : public std::exception
{
public:

	SocketException(){}


private:

};


class SendMessageException : public std::exception{};
class RecvMessageException : public std::exception{};


#endif // !SOCKET_H

