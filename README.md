# Zest: non-uniform random variate generator using the Generalized Ziggurat algorithm

Zest implements the [generalized Ziggurat] algorithm which generalizes the [original Ziggurat] invented by Marsaglia and Tsang to unimodal and unbounded monotone probability density functions with possibly infinite support. This library supports the normal, Cauchy, exponential, gamma, Weibull, log-normal, student's t and Fisher's f distributions. It is faster than both Standard Template Library (STL) and Boost in most cases, sometimes by a factor of more than 10.

## How fast is it?

This tables shows the (approximate) relative speed of Zest (with 1024 regions) to STL and Boost.

| Distribution           | STL   | Boost |
|------------------------|-------|-------|
| normal                 | 3.6   | 1.0   |
| log-normal             | 3.1   | 1.0   |
| Cauchy                 | 4     | 4     |
| exponential            | 3.3   | 0.67  |
| gamma α>1              | 2.5   | 7     |
| gamma 0.1<α<1          | 5     | 2.4-5 |
| Weibull α>1            | 3.3   | 3.3   |
| Weibull 0.1<α<1        | 2-4   | 2-4   |
| student's t ν>1        | 10    | 20    |
| student's t 0.1<ν<1    | 6-16  | 2.5-8 |
| Fisher's f d1,d2>1     | 6     | 11-14 |
| Fisher's f 0.2<d1,d2<1 | 2-6   | 1.3-7 |

## How to install it?

There's no installation required. There's only a single header file `zest.hpp` and you can either copy 
that into your source directory or copy it into the system's include path.

## How to use it?

Generating random numbers from predefined distributions is covered in the basic usage, while defining and using your own probability density function is covered in the advanced usage.

### Basic usage

Please see our [example](example.cpp). It is fairly self-explanatory.

`Ziggurat` is a class template with four template parameters:
 1. `class Distribution`: Self-explanatory
 2. `class URBG`: Uniform Random Bit Generator class  
  It must generate unsigned integer random numbers in [0,2<sup>32</sup>) or [0,2<sup>64</sup>).
 3. `uint_fast16_t N`: Number of regions (optional, default is 1024)
  Must be an integer power of two.
 4. `typename float_type`: type of the floating-point numbers to use (optional, default is `double`)  
  Usually `float` or `double`

### Advanced usage

You can use a custom probability density function by defining it as a class and supplying it as the `class Distribution` template parameter to the Zest. The current [generalized Ziggurat] algorithm only supports unimodal distributions, that is distributions that have no more than one local maxima in their PDF.

We describe the syntax for the needed class here. Looking at the Zest predefined distribution classes can also be very helpful.

The following members are always mandatory:
 - `float_type mode`: floating-point number denoting the mode of the distribution.
 - `bool is_mode_unbounded`: boolean denoting whether the density is unbounded at the mode or not.
 - `static zest::detail::DistCategory dist_category`: enum class describing the type of the distribution. Possible values are:
   + `ASYMMETRIC`
   + `SYMMETRIC`
   + `STRICTLY_INCREASING` for monotone PDFs with positive derivative
   + `STRICTLY_DECREASING` for monotone PDFs with negative derivative

`ASYMMETRIC` distributions must define two nested classes, `Left` and `Right` which will represent the monotonic distributions on either side of the mode. The `dist_category` must be `STRICTLY_INCREASING` for the `Left` and `STRICTLY_DECREASING` for the `Right`, respectively.

`SYMMETRIC`, `STRICTLY_INCREASING` and `STRICTLY_DECREASING` must define the following mandatory members:
 - `float_type pdf (float_type x)`: function computing the value of PDF at `x`. Need not be normalized.
 - `float_type strip_area (float_type x)`: function computing the area of a horizontal strip intersecting the PDF at `x`, as described in the [generalized Ziggurat] algorithm.
 - `static zest::detail::TailCategory tail_category`: enum class describing the type of distribution's tail algorithm. Possible values are:
   + `FINITE` for finite tails. The following members are mandatory:
     * `float_type support`: floating-point number denoting the endpoint of the tail's support (beyond which the PDF is always zero).
   + `MAP` for a mapping algorithm (without rejection sampling). The following members are mandatory:
     * `float_type tail_value_rel_mode (float_type tail_start_rel_mode, float_type u)`: mapping function for a tail starting at `tail_start_rel_mode` (relative to the mode) that maps a uniform variate `u` into the tail distribution.
   + `MAP_REJECT` for a mapping algorithm with rejection sampling. The following members are mandatory:
     * `float_type tail_value_rel_mode (float_type tail_start_rel_mode, float_type u)`: mapping function for a tail starting at `tail_start_rel_mode` (relative to the mode) that maps a uniform variate `u` into the tail distribution.
     * `float_type tail_probability (float_type tail_start_rel_mode, float_type x_rel_mode)` function computing acceptance probability of `x_rel_mode` (relative to the mode) for a tail distribution starting at `tail_start_rel_mode`. Note that `x_rel_mode` is the transformed variate, not the initial uniform variate `u`.

All mandatory members must be `public`. All mandatory member variables and functions must be either `const` or `static` except `dist_category` and `tail_category` which must always be `static`. If all mandatory members are defined as `static` members (like `StandardNormal`), the `Ziggurat` constructor should be called with no arguments. Otherwise an instance of the distribution's class must be provided to the `Ziggurat` constructor. `ASYMMETRIC` distributions having non-`static` members, must have instances of `Left` and `Right` classes named `left` and `right`, respectively. That is they must have the following members `Left left` and `Right right`. Please see the code of `Gamma`, `Weibull`, `LogNormal` or `FisherF` distributions as an example.

[original Ziggurat]: to-be-filled:-link-to-Ziggurat
[generalized Ziggurat]: to-be-filled:-link-to-draft
