% The following function uses the squared Pearson's correlation coefficient
% to measure the similarity between vectors x and y
function r = similarityMeasure(x, y)
mean_x = mean(x);
mean_y = mean(y);
y_difference = y - mean_y;
x_difference = x - mean_x;
mul_x_difference_with_y_difference = x_difference.*y_difference;
x_square = x_difference.^2;
y_square = y_difference.^2;
sum_product = sum(mul_x_difference_with_y_difference);
sum_x_square = sum(x_square);
sum_y_square = sum(y_square);
product_x_y_square = sum_x_square*sum_y_square;
denom = sqrt(product_x_y_square);
r = sum_product / denom;
r = r^2;

