#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <iterator>
#include <sys/types.h>
#include <dirent.h>

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& c)
{
  os << "{ ";
  std::copy(
      std::begin(c),
      std::end(c),
      std::ostream_iterator<T>{os, " "}
  );
  os << '}';

  return os;
}

std::vector<std::string> entries(DIR* dir)
{
  std::vector<std::string> result;

  dirent entry;
  for (auto* r = &entry; readdir_r(dir, &entry, &r) == 0 && r; ) {
    // here `entry.d_name` is the name of the current entry
    result.emplace_back(entry.d_name);
  }
 
  return result;
}

auto my_make_unique(const char* name){
  return std::unique_ptr<DIR,void(*)(DIR* p)>(opendir(name),[](DIR* p) { 
    closedir(p);
    std::cout<< "closedir was called\n";
  });
}


int main(int argc, char* argv[])
{
  std::string const name = argc > 1 ? argv[1] : ".";

  // struct dirDeleter {
  //   void operator()(DIR* p){
  //     std::cout << "dirDeleter has been called\n";
  //     closedir(p);
  //   }
  // };
  //auto dirDeleter = [](DIR* p){ closedir(p); };

  std::cout << "creating unique_ptr\n";

  //auto pdir = std::unique_ptr<DIR,dirDeleter>(opendir(name.c_str()));
  //auto pdir = std::unique_ptr<DIR,void(*)(DIR*)>(opendir(name.c_str()),[](DIR* p) { closedir(p);});
  //auto pdir = std::make_unique<DIR,void(*)(DIR*)>(opendir(name.c_str()),[](DIR* p) { closedir(p);});
  // auto pdir = std::shared_ptr<DIR>(opendir(name.c_str()),[](DIR* p){
  //   std::cout << "Calling closedir\n"; 
  //   closedir(p);
  // });

  // function pointers are declared like this:
  // return_type(*pointer_name)(parameter_type1,parameter_type2,parameter_type3) 
  // and then it can be defined using:
  // pointer_name = &function
  // auto pdir = std::unique_ptr<DIR,void(*)(DIR*)>(opendir(name.c_str()),[](DIR* p){
  //   closedir(p);
  // });

  auto pdir = my_make_unique(name.c_str());


  //std::function<int(DIR*,int)>
  // create a smart pointer to a DIR here, with a deleter
  // relevant functions and data structures are
  // DIR* opendir(const char* name);
  // int  closedir(DIR* dirp);

  std::cout << "Calling entries()\n";
  std::vector<std::string> v = entries(pdir.get());
  std::cout << v << '\n';

  std::vector<std::unique_ptr<FILE>> vFiles;



  std::cout << "About to end the program\n";
}
