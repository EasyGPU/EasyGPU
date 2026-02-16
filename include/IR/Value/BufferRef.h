/**
 * BufferRef.h:
 *      @Descripiton    :   The buffer reference for DSL access - Full IR integration
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 * 
 * Design inspired by VarVector.h swizzle access:
 * - BufferElement<T> acts like a variable that can be read/written
 * - Implicit conversion to Var<T> for reading
 * - operator= for writing back to buffer
 * - Full Expr<T> integration via operator overloading
 */
#ifndef EASYGPU_BUFFERREF_H
#define EASYGPU_BUFFERREF_H

#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <format>

// Forward declaration
namespace GPU::Runtime {
    template<typename T>
    class Buffer;
    enum class BufferMode;
}

namespace GPU::IR::Value {
    // Forward declaration for element type
    template<typename T>
    class BufferElement;

    /**
     * The buffer reference class for DSL access
     * Usage:
     *   auto buf = buffer.Bind();
     *   Var<float> v = buf[id];        // Read
     *   buf[id] = value;               // Write
     *   buf[id] = buf[i] * 2.0f;       // Expression
     * 
     * @tparam T The element type of the buffer
     */
    template<typename T>
    class BufferRef {
    public:
        BufferRef(std::string bufferName, uint32_t binding)
            : _bufferName(std::move(bufferName))
            , _binding(binding) {
        }

        [[nodiscard]] uint32_t GetBinding() const { return _binding; }
        [[nodiscard]] const std::string& GetBufferName() const { return _bufferName; }

        /**
         * Array access - returns a BufferElement that can be read/written
         */
        [[nodiscard]] Var<T> operator[](const Var<int>& index) const;
        [[nodiscard]] Var<T> operator[](const Expr<int>& index) const;
        [[nodiscard]] Var<T> operator[](int index) const;

    private:
        std::string _bufferName;
        uint32_t _binding;
    };

    // =============================================================================
    // Implementation of BufferRef::operator[]
    // =============================================================================

    template<typename T>
    [[nodiscard]] Var<T> BufferRef<T>::operator[](const Var<int>& index) const {
        return Var<T>(std::format("{}[{}]", GetBufferName(), Builder::Builder::Get().BuildNode(*index.Load().get())));
    }

    template<typename T>
    [[nodiscard]] Var<T> BufferRef<T>::operator[](const Expr<int>& index) const {
        return Var<T>(std::format("{}[{}]", GetBufferName(), Builder::Builder::Get().BuildNode(*index.Node())));
    }

    template<typename T>
    [[nodiscard]] Var<T> BufferRef<T>::operator[](int index) const {
        return Var<T>(std::format("{}[{}]", GetBufferName(), std::to_string(index)));
    }

    /**
     * Type alias for convenience
     */
    template<typename T>
    using buffer = BufferRef<T>;

} // namespace GPU::IR::Value

#endif //EASYGPU_BUFFERREF_H
