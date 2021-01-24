#include <iostream>
using namespace std;

void doStuff(const char name[], int b) {
  cout << "doing stuff with " << name << " and " << b << endl;
}

#define doStuff(params) doStuff(__PRETTY_FUNCTION__, params)

class Foobar {
public:
  static void Method(const char name[], int b) {
    cout << "doing MAGIC with " << name << " and " << b << endl;
  }
};

template <typename FUNC>
void doMagic()

#define Method(params) Foobar::Method(__PRETTY_FUNCTION__, params)

int main() {
  cout << "Hello world!" << endl;
  cout << "Here I am:" << endl;
  cout << __FUNCTION__ << endl;
  cout << __PRETTY_FUNCTION__ << endl;

  doStuff(5);
  Foobar::Method(32);

  cout << endl << endl;
}