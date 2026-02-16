/**
 * Value.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <IR/Value/Value.h>

namespace GPU::IR::Value {
    Value::Value(Value &&other) noexcept : _node(std::move(other._node)) {}
    
    Value &Value::operator=(Value &&other) noexcept {
        if (this != &other) {
            _node = std::move(other._node);
        }
        return *this;
    }
}