# Calculus Project

This project covers various topics in calculus, including summation and product notations, derivatives, integrals, and partial derivatives. The goal is to understand and apply these concepts through multiple-choice questions and Python scripts.

## Resources

Read or watch:

- [Sigma Notation (starting at 0:32)](https://example.com/sigma-notation)
- [Π Product Notation (up to 0:20)](https://example.com/pi-notation)
- [Sigma and Pi Notation](https://example.com/sigma-pi-notation)
- [What is a Series?](https://example.com/what-is-a-series)
- [What is a Mathematical Series?](https://example.com/mathematical-series)
- [List of mathematical series: Sums of powers](https://example.com/sums-of-powers)
- [Bernoulli Numbers(Bn)](https://example.com/bernoulli-numbers)
- [Bernoulli Polynomials(Bn(x))](https://example.com/bernoulli-polynomials)
- [Derivative (mathematics)](https://example.com/derivative)
- [Calculus for ML](https://example.com/calculus-for-ml)
- [1 of 2: Seeing the big picture](https://example.com/seeing-the-big-picture)
- [2 of 2: First Principles](https://example.com/first-principles)
- [1 of 2: Finding the Derivative](https://example.com/finding-the-derivative)
- [2 of 2: What do we discover?](https://example.com/what-do-we-discover)
- [Deriving a Rule for Differentiating Powers of x](https://example.com/differentiating-powers-of-x)
- [1 of 3: Introducing a substitution](https://example.com/introducing-substitution)
- [2 of 3: Combining derivatives](https://example.com/combining-derivatives)
- [How To Understand Derivatives: The Product, Power & Chain Rules](https://example.com/understand-derivatives)
- [Product Rule](https://example.com/product-rule)
- [Common Derivatives and Integrals](https://example.com/common-derivatives-integrals)
- [Introduction to partial derivatives](https://example.com/introduction-to-partial-derivatives)
- [Partial derivatives - How to solve?](https://example.com/partial-derivatives-solve)
- [Integral](https://example.com/integral)
- [Integration and the fundamental theorem of calculus](https://example.com/integration-fundamental-theorem)
- [Introduction to Integration](https://example.com/introduction-to-integration)
- [Indefinite Integral - Basic Integration Rules, Problems, Formulas, Trig Functions, Calculus](https://example.com/indefinite-integral)
- [Definite Integrals](https://example.com/definite-integrals)
- [Definite Integral](https://example.com/definite-integral)
- [Multiple integral](https://example.com/multiple-integral)
- [Double integral 1](https://example.com/double-integral-1)
- [Double integrals 2](https://example.com/double-integrals-2)

## Learning Objectives

By the end of this project, you should be able to explain the following concepts:

- Summation and Product notation
- What is a series?
- Common series
- What is a derivative?
- What is the product rule?
- What is the chain rule?
- Common derivative rules
- What is a partial derivative?
- What is an indefinite integral?
- What is a definite integral?
- What is a double integral?

## Requirements

### Multiple Choice Questions

- Allowed editors: vi, vim, emacs
- Type the number of the correct answer in your answer file
- All your files should end with a new line

Example:

```bash
What is 9 squared?
1. 99
2. 81
3. 3
4. 18
alexa@ubuntu$ cat answer_file
2
alexa@ubuntu$

Python Scripts
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module
All your files must be executable
The length of your files will be tested using wc
Tasks
0. Sigma is for Sum
File: 0-sigma_is_for_sum

1. The Greeks pronounce it sEEgma
File: 1-seegma

2. Pi is for Product
File: 2-pi_is_for_product

3. The Greeks pronounce it pEE
File: 3-pee

4. Hello, derivatives!
File: 4-hello_derivatives

5. A log on the fire
File: 5-log_on_fire

6. It is difficult to free fools from the chains they revere
File: 6-voltaire

7. Partial truths are often more insidious than total falsehoods
File: 7-partial_truths

8. Put it all together and what do you get?
File: 8-all-together

9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities
File: 9-sum_total.py

10. Derive happiness in oneself from a good day's work
File: 10-matisse.py

11. Good grooming is integral and impeccable style is a must
File: 11-integral

12. We are all an integral part of the web of life
File: 12-integral

13. Create a definite plan for carrying out your desire and begin at once
File: 13-definite

14. My talents fall within definite limitations
File: 14-definite

15. Winners are people with definite purpose in life
File: 15-definite

16. Double whammy
File: 16-double

17. Integrate
File: 17-integrate.py

Write a function def poly_integral(poly, C=0): that calculates the integral of a polynomial:

poly is a list of coefficients representing a polynomial
The index of the list represents the power of x that the coefficient belongs to
Example: if f(x) = x^3 + 3x + 5, poly is equal to [5, 3, 0, 1]
C is an integer representing the integration constant
If a coefficient is a whole number, it should be represented as an integer
If poly or C are not valid, return None
Return a new list of coefficients representing the integral of the polynomial
The returned list should be as small as possible
Example:

bash
Copier le code
alexa@ubuntu:calculus$ cat 17-main.py
#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
alexa@ubuntu:calculus$ ./17-main.py
[0, 5, 1.5, 0, 0.25]
Repo
GitHub repository: holbertonschool-machine_learning
Directory: math/calculus
