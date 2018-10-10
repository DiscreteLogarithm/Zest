# Brief guide to the source code for reviewers and developers

Design decisions are taken according to these three design ideals:
1. The user should only see and use a unified user interface irrespective of whether s/he is dealing with a 
   monotonic or symmetric or asymmetric, bounded or unbounded, etc distribution.
2. Runtime should not be wasted.
3. The user should be able to define and implement its own distribution in the same manner as the predefined 
   distributions without modifying Zest.

Run-time would be wasted if each time we want a new random variate, we had to check whether a distribution is  
monotonic or symmetric or asymmetric, bounded or unbounded, etc.
This dictates separate implementation for each case.
We also want a unified interface, but using virtual functions also wastes runtime.
Therefore we use template inheritance, a technique in which an interface class inherits from one of many 
implementation classes depending on its template parameter(s).
Each distribution is defined as a class and will be used as template parameter to the `Ziggurat` class.
This template parameter determines which implementation class should be used at compile-time.

## namespaces

Everything except the `Ziggurat` class is wrapped inside the `zest` namespaces. There's two namespace nested
in the `zest` namespace: `math` (contains some math functions) and `detail`. Anything inside the `detail` 
namespace is considered to be implementation's detail and is not intended to be directly used by the user.
All predefined distribution classes are out of the `detail` namespace.
There's a `type_utils` namespace inside the `detail`, which contains utilities for checking whether the 
supplied distribution class has all the required members and selector classes which select which 
implementation class the `Ziggurat` class should inherit from.
Implementation classes are organized in a hierarchy of namespaces inside the `detail` namespace.
At the first level of the hierarchy, there are `symmetric`, `monotonic`, and `asymmetric` namespaces, 
determining the type of the distribution.
`symmetric` and `monotonic` namespaces each have two namespaces nested inside them (second level):
`singular` and `nonsingular` determining whether the distribution's density is unbounded at the mode or not.
At the third level there are three namespaces - `finite`, `map`, `map_reject` - nested inside the `singular` 
and `nonsingular` namespaces. `finite` contains Ziggurat implementations for distributions with finite tails. 
`map` contains Ziggurat implementations for distributions that need a mapping function to generate a random 
number from their (infinite) tail distribution. Similarly, `map_reject` contains Ziggurat implementations for 
distributions that need both a mapping function and an acceptance probability function to generate a random 
number from the (infinite) tail distribution using rejection sampling.
Implementation classes inside the `asymmetric` use those ones declared in the `monotonic` namespace. 
Therefore, no namespace is declared within `asymmetric`.

## `static_impl` vs `impl`

If all required members of a distribution class are static, no instance of that class is required
to evaluate the mode, pdf, etc and constructing an instance of that distribution is unnecessary.
`static_impl` classes' constructors do not take a distribution instance argument,
while `impl` classes's constructors require a distribution instance argument.
`static_impl` classes use the call syntax of static members (for example `Dist::pdf (x)` where `Dist` is a 
typename), while `impl` classes use call the member functions the usual way (for example `dist.pdf (x)` where 
`dist` is an instance of `Dist`).

## type selection

Compile-time type selection could be achieved by template specialization of a class with a member type.
Suppose we only had 3 implementation classes for the `symmetric`, `monotonic`, and `asymmetric` cases.
As detailed in the [README] file, all distribution classes must have an static member of enum class type 
`DistCategory` named `dist_category` determining the type of the distribution. This example demonstrates how 
`dist_category` could be used to select between implementation classes at compile-time:

```C++
enum class DistCategory {STRICTLY_DECREASING, STRICTLY_INCREASING, SYMMETRIC, ASYMMETRIC};

template <DistCategory category>
struct selector {
  using type = general_impl;      // general_impl is an empty class because
                                  // we only have 3 implementations for the 3 cases
};

template <>
struct selector<DistCategory::STRICTLY_DECREASING> {
  using type = monotonic_impl;
};

template <>
struct selector<DistCategory::STRICTLY_INCREASING> {
  using type = monotonic_impl;
};

template <>
struct selector<DistCategory::SYMMETRIC> {
  using type = symmetric_impl;
};

template <>
struct selector<DistCategory::ASYMMETRIC> {
  using type = asymmetric_impl;
};

template <class Distribution>
class Ziggurat : public typename selector<Distribution::dist_category>::type {};
```

There is no virtual function overriding here because it is the interface that is inheriting from the 
implementation and not the other way around. There is not a single virtual function in Zest and therefore all 
objects would be vtable free which improves performance.

The actual selector is more complicated because the decision tree is more complex and have more than one 
level.
To understand the decision tree, one should review the Advanced usage section of the [README] and the namespaces section of this file.

## type verification

Zest leverages SFINAE (substitution failure is not an error) rule to assert if a distribution has all the 
required members and whether all of them are static or not. Here's a small example:

```C++
struct substitution_failure {};

template <class Dist>
class has_pdf {
  template <class X>
  static auto check_pdf (const X &x) -> decltype (x.pdf (double{}));
  static substitution_failure check_pdf (...);
 public:
  static constexpr bool value = std::is_floating_point<decltype(check_pdf (std::declval<Dist>()))>::value;
};
```

The value of `has_pdf<Dist>::value` would indicate whether `Dist` has a const member function named pdf taking 
a `double` argument (or a type implicitly convertible from `double`) and returning a floating-point number.
This works because overload resolution always selects the ellipsis overload (`check_pdf (...)`) last and if 
the compiler fails to determine the return type for the template overload, no error will be emitted and it
will simply be ignored (and the ellipsis overload will be used instead).
Because the return type is declared to be the same as the return type of `x.pdf (double{})` called on a const instance `x`, it can only be successfully determined if there is a const member function `pdf` taking a `double` argument.

To check whether a class has a static member the return type should be specified as `decltype (X::pdf (double{}))`.

[README]: README.md
