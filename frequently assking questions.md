# Interview Questions on KNN

1. **How does the KNN algorithm work?**  
   It classifies based on majority vote from K nearest neighbors.

2. **How do you choose the right K?**  
   By testing different values; usually using cross-validation.

3. **Why is normalization important in KNN?**  
   Because distance metrics are scale-sensitive.

4. **What is the time complexity of KNN?**  
   O(n) per query, since it computes distances to all points.

5. **What are pros and cons of KNN?**  
   Pros: Simple, effective. Cons: Slow with large data, memory-intensive.

6. **Is KNN sensitive to noise?**  
   Yes, noisy data can mislead nearest neighbor classification.

7. **How does KNN handle multi-class problems?**  
   By majority voting across all classes.

8. **What’s the role of distance metrics in KNN?**  
   Distance metrics (Euclidean, Manhattan, etc.) decide neighbor closeness.
