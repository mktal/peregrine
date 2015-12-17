/*
* @Author: Xiaocheng Tang
* @Date:   2015-12-15 16:09:52
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2015-12-15 21:12:41
*/

#ifndef __LHAC__Array__
#define __LHAC__Array__

#include "example.h"

template <typename TypeValue>
class Array {
public:
    Array(size_t size) : _size(size) {
        std::cout << "copy" << std::endl;
        m_data = new TypeValue[_size];
        memset(m_data, 0, sizeof(TypeValue) * _size);
    }

    Array(TypeValue* data, size_t size) : m_data(data), _size(size) {
        std::cout << "no copy" << std::endl;
        _owner = false;
    }

    Array(const Array &s) : _size(s._size) {
        m_data = new TypeValue[_size];
        memcpy(m_data, s.m_data, sizeof(TypeValue) * _size);
    }

    Array(Array &&s) : _size(s._size), m_data(s.m_data) {
        std::cout << "move" << std::endl;
        s._size = 0;
        s.m_data = nullptr;
    }

    ~Array() {
        if (_owner) {
            delete[] m_data;
        }
        else
            std::cout << "not freeing" << std::endl;
    }

    Array &operator=(const Array &s) {
        delete[] m_data;
        _size = s._size;
        m_data = new TypeValue[_size];
        memcpy(m_data, s.m_data, sizeof(TypeValue) * _size);
        return *this;
    }

    Array &operator=(Array &&s) {
        if (&s != this) {
            delete[] m_data;
            _size = s._size; m_data = s.m_data;
            s._size = 0; s.m_data = nullptr;
        }
        return *this;
    }

    TypeValue operator()(size_t i) const {
        return m_data[i];
    }

    TypeValue &operator()(size_t i) {
        return m_data[i];
    }

    TypeValue *data() { return m_data; }

    size_t size() const { return _size; }
private:
    size_t _size;
    bool _owner = true;
    TypeValue *m_data;
};

#endif /* defined(__LHAC__Array__) */
