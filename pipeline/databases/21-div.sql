-- Creates a function SafeDiv that divides (and returns) the first by the second number
--    or returns 0 if the second number is 0
-- Function SafeDiv takes 2 arguments:
--    a, INT
--    b, INT

DELIMITER //

CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
        IF b = 0 THEN
           RETURN 0;
        ELSE
                RETURN (a / b);
        END IF;
END //
DELIMITER ;
