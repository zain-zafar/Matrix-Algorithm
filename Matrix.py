class MatrixIndexError(Exception):
    '''An attempt has been made to access an invalid index in this matrix'''
    pass


class MatrixDimensionError(Exception):
    '''An attempt has been made to perform an operation on this matrix which
    is not valid given its dimensions'''
    pass


class MatrixInvalidOperationError(Exception):
    '''An attempt was made to perform an operation on this matrix which is
    not valid given its type'''
    pass


class MatrixNode():
    '''A general node class for a matrix'''

    def __init__(self, contents, right=None, down=None):
        '''(MatrixNode, obj, MatrixNode, MatrixNode) -> NoneType
        Create a new node holding contents, that is linked to right
        and down in a matrix
        '''
        self._contents = contents
        self._right = right
        self._down = down

    def __str__(self):
        '''(MatrixNode) -> str
        Return the string representation of this node
        '''
        return str(self._contents)

    def get_contents(self):
        '''(MatrixNode) -> obj
        Return the contents of this node
        '''
        return self._contents

    def set_contents(self, new_contents):
        '''(MatrixNode, obj) -> NoneType
        Set the contents of this node to new_contents
        '''
        self._contents = new_contents

    def get_right(self):
        '''(MatrixNode) -> MatrixNode
        Return the node to the right of this one
        '''
        return self._right

    def set_right(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set the new_node to be to the right of this one in the matrix
        '''
        self._right = new_node

    def get_down(self):
        '''(MatrixNode) -> MatrixNode
        Return the node below this one
        '''
        return self._down

    def set_down(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
        Set new_node to be below this one in the matrix
        '''
        self._down = new_node


class Matrix():
    '''A class to represent a mathematical matrix'''

    def __init__(self, m, n, default=0):
            self._head = MatrixNode(None)
            self._rows = m
            self._columns = n
            self._default = default
            curr_node = self._head

            # loop through the number of desired rows for the matrix, creating
            # nodes starting from 0 and upto the number of rows, and setting
            # each node below the other, to link them in a top-bottom row.
            for i in range(m):
                # a row node is created
                row = MatrixNode(i)
                # that row matrix is set below the current node, which is None
                curr_node.set_down(row)
                # change the current node to the node below it, this allows
                # for linking the nodes from None to the last needed node in
                # a row
                curr_node = curr_node.get_down()

            # Reset the current to be the head Node, which is None
            curr_node = self._head

            # Loop through the number of columns needed to be made in the
            # matrix, starting from 0 to n which is the number of columns.
            for j in range(n):
                # a column node is created, which is set to the right of
                # the head node, which is None
                col = MatrixNode(j)
                curr_node.set_right(col)
                # Set the current node as the node to the right of it, this
                # allows us to make number of columns needed in matrix as
                # linked node's
                curr_node = curr_node.get_right()

            # initialize a node which acts as the start position of the nodes
            # which lie inside the already created row and column.
            start_node = self._head
            # set current node as the start node
            curr_node = start_node
            # loop through the number of rows starting from 0, and every time
            # this loop repeats, the current node right moves to the next
            # right node and the current node down becomes the node below it.
            # This allow to keep placing new nodes in each row by linking
            for i in range(m):
                # set current node right as the node to the right
                # of the start node
                curr_node_right = start_node.get_right()
                # set current node down as the node below the start node
                curr_node_down = start_node.get_down()
                start_node = curr_node_down
                # Loop through the columns from 0
                for j in range(n):
                    # create a new node which will be placed inside the
                    # already created row and column. The new node
                    # will contain the data/content of the default value
                    new_node = MatrixNode(default)
                    # Set the new node below current right node
                    curr_node_right.set_down(new_node)
                    # set the new node to the right of current node down.
                    curr_node_down.set_right(new_node)
                    # after inserting the new node, change current node right
                    # to the node next to it and current node down to the
                    # newly created node, this allows for placing nodes in
                    # all of the rows.
                    curr_node_right = curr_node_right.get_right()
                    curr_node_down = new_node

    def goto_row_column_node(self, i, j):
        '''(Matrix, int, int) -> MatrixNode

        REQ: i and j are valid index values of the Matrix

        Return a node at m[i,j] location.
        '''
        current_node = self._head
        # if i and j are index values which lie in the Matrix, continue
        if i < self._rows and j < self._columns:
            # Starting at the current node, move to the right until
            # j is reached
            for count in range(j + 1):
                # keep moving to the right, until the j'th column
                # is not reached
                current_node = current_node.get_right()
            # Starting at the current, which will be at j'th column and 0'th
            # row, keep moving down until the i'th row is reached
            for count in range(i + 1):
                current_node = current_node.get_down()
            # Now that the (i, j) node has been reached, return the value
            # of that node
            return current_node
        # if i and j are not valid index ranges inside the Matrix, raise
        # MatrixIndexError exception
        else:
            raise MatrixIndexError

    def get_val(self, i, j):
        '''(Matrix, int, int) -> float
        Return the value of m[i,j] for this matrix m
        '''
        # find the m[i,j] node and get it equal to node_to_find
        node_to_find = self.goto_row_column_node(i, j)
        # return the contents of the found desired node
        return node_to_find.get_contents()

    def set_val(self, i, j, new_val):
        '''(Matrix, int, int, float) -> NoneType
        Set the value of m[i,j] to new_val for this matrix m
        '''
        # find the node at m[i,j] using helper function goto_row_column_node
        # once the value of the node is found, store it
        node_to_be_found = self.goto_row_column_node(i, j)
        # set the found node's content equal to new_val
        node_to_be_found.set_contents(new_val)

    def get_row(self, row_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the row_num'th row of this matrix
        '''
        # if the row_num is less than the row index of Matrix, continue
        if row_num < self._rows and row_num >= 0:
            # create a one dimensional matrix, which is made to store all the
            # node column values of row_num
            store_row = OneDimensionalMatrix(1, self._columns, self._default)
            # Set current node as the head node
            current_node = self._head
            # Loop through until the desired row_num index is reached, by
            # going down each node from head node
            for count in range(row_num + 1):
                current_node = current_node.get_down()
            # Once the row desired is reached, then store every Node to the
            # right of current node, one at a time, in the 1D Matrix
            for count in range(self._columns):
                current_node = current_node.get_right()
                # Store nodes which are to the right of current node, in
                # the 1D Matrix
                store_row.set_item(row_num, current_node.get_contents())
            # Return the 1D matrix
            return store_row
        # Raise an exception error, iff row_num does not meet the requirements
        else:
            raise MatrixIndexError

    def set_row(self, row_num, new_row):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the row_num'th row of this matrix to those of new_row
        '''
        # as long as the row number is less than the total number of columns
        # inside the Matrix and also as long as the columns of the 1D Matrix
        # is equal to the number of columns in Matrix, then continue
        if row_num < new_row._columns and new_row._columns == self._columns:
            # loop through each 1D Matrix node
            for count in range(new_row._columns):
                # for each node, get the node value
                value_of_node = new_row.get_item(count)
                # Make the Matrix element that lies on the same m[i,j] as the
                # 1D m[i,j] equal to the 1D node's value
                self.set_val(row_num, count, value_of_node)
        else:
            raise MatrixIndexError

    def get_col(self, col_num):
        '''(Matrix, int) -> OneDimensionalMatrix
        Return the col_num'th column of this matrix
        '''
        # if the col_num given is less than the number of columns in the Matrix
        # then continue, since the col_num lies inside Matrix
        if col_num < self._columns and col_num >= 0:
            # Create a one dimensional matrix which will create a Matrix to
            # store the row values at the given column.
            store_column = OneDimensionalMatrix(self._rows, 1, self._default)
            # set the current node as the head node of the matrix
            current_node = self._head
            # starting from the current node, keep moving right until the
            # needed column is found
            for count in range(col_num + 1):
                current_node = current_node.get_right()
            # Once the col_num is reached, go down each node until the current
            # node reaches the end of the row, while storing each node's
            # values in the 1D Matrix.
            for count in range(self._rows):
                current_node = current_node.get_down()
                store_column.set_item(col_num, current_node.get_contents())
            # Return the 1D Matrix
            return store_column
        # If col num does not meet the requirement, raise MatrixIndexError
        # Exception
        else:
            raise MatrixIndexError

    def set_col(self, col_num, new_col):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
        Set the value of the col_num'th column of this matrix to those
        of new_row
        '''
        # as long as the row number is less than the total number of columns
        # inside the Matrix and also as long as the columns of the 1D Matrix
        # is equal to the number of columns in Matrix, then continue
        if col_num < new_col._rows and new_col._rows == self._rows:
            # loop through each 1D Matrix node
            for count in range(new_col._rows):
                # for each node, get the node value
                value_of_node = new_col.get_item(count)
                # Make the Matrix element that lies on the same m[i,j] as the
                # 1D m[i,j] equal to the 1D node's value
                self.set_val(count, col_num, value_of_node)
        else:
            raise MatrixIndexError

    def swap_rows(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of rows i and j in this matrix
        '''
        # if index i and index j are less than the rows of Matrix, continue
        # columns of the matrix, then continue
        if i < self._rows and j < self._rows:
            # Get the first row values
            first_row_swap = self.get_row(i)
            # Get the second row values
            second_row_swap = self.get_row(j)
            # Set the second row value as first row value
            self.set_row(i, second_row_swap)
            # Set the second row value as first row value
            self.set_row(j, first_row_swap)

        else:
            # if condition is not met, raise MatrixIndexError exception
            MatrixIndexError

    def swap_cols(self, i, j):
        '''(Matrix, int, int) -> NoneType
        Swap the values of columns i and j in this matrix
        '''
        # if any errors rise, catch them then raise exception
        try:
            # Get the first column values
            first_column_swap = self.get_col(i)
            # Get the second column values
            second_column_swap = self.get_col(j)
            # Set the second row value as first row value
            self.set_row(i, second_column_swap)
            # Set the second row value as first row value
            self.set_row(j, first_column_swap)

        except:
            # raise MatrixIndexError exception
            raise MatrixIndexError

    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
        Increase all values in this matrix by add_value
        '''
        # loop through each row starting from the 0'th row all the way to the
        # last row
        for count in range(self._rows):
            # loop thorugh each column starting from the 0'th all the way to
            # last column
            for counter in range(self._columns):
                # find the value of the node at m[count, counter]
                value_of_node = self.get_val(count, counter)
                # now add the value of the node to the given add_value
                # and reset each nodes value as the sum of value of node and
                # the given value to add
                self.set_val(count, counter, value_of_node + add_value)

    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
        Decrease all values in this matrix by sub_value
        '''
        # set the current node as none
        current_node = self._head
        # loop through each row starting from the 0'th row all the way to the
        # last row
        for count in range(self._rows):
            # loop thorugh each column starting from the 0'th all the way to
            # last column
            for counter in range(self._columns):
                # find the value of the node at m[count, counter]
                value_of_node = self.get_val(count, counter)
                # now subtract the value of the node by the given sub_value
                # and reset each nodes value as the subtraction of
                # the value of node and the given value
                self.set_val(count, counter, value_of_node - sub_value)

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
        Multiply all values in this matrix by mult_value
        '''
        # set the current node as head node
        current_node = self._head
        # loop through each row starting from the 0'th row all the way to the
        # last row
        for count in range(self._rows):
            # loop thorugh each column starting from the 0'th all the way to
            # last column
            for counter in range(self._columns):
                # find the value of the node at m[count, counter]
                value_of_node = self.get_val(count, counter)
                # now multiply the value of the node to the mult_value
                # and reset each nodes value as the multiplication
                # of value of node and the given value to multiply
                self.set_val(count, counter, value_of_node * mult_value)

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the sum of this matrix and adder_matrix
        '''

    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
        Return a new matrix that is the product of this matrix and mult_matrix
        '''


class OneDimensionalMatrix(Matrix):
    '''A 1xn or nx1 matrix.
    (For the purposes of multiplication, we assume it's 1xn)'''
    # check to see if the matrix is 1 dimentional:

    def get_item(self, i):
        '''(OneDimensionalMatrix, int) -> float
        Return the i'th item in this matrix
        '''
        # check to see if it is a 1xn matrix
        if self._rows == 1 and self._columns > 1:
            # if it is, then get the value of node at location m[1,n]
            return self.get_val(0, i)

        # check to see if it is a nx1
        elif self._rows > 1 and self._columns == 1:
            # if it is a nx1 matrix, return node value at m[n,1]
            return self.get_val(i, 0)

        # if it is not a 1D matrix
        else:
            raise MatrixDimensionError

    def set_item(self, i, new_val):
        '''(OneDimensionalMatrix, int, float) -> NoneType
        Set the i'th item in this matrix to new_val
        '''
        # check to see if it is a 1xn matrix
        if self._rows == 1 and self._columns > 1:
            # if it is, then get the value of node at location m[1,n]
            return self.set_val(0, i, new_val)

        # check to see if it is a nx1
        elif self._rows > 1 and self._columns == 1:
            # if it is a nx1 matrix, return node value at m[n,1]
            return self.set_val(i, 0, new_val)

        # if it is not a 1D matrix, raise a MatrixDimensionError
        else:
            raise MatrixDimensionError


class SquareMatrix(Matrix):
    '''A matrix where the number of rows and columns are equal'''

    def transpose(self):
        '''(SquareMatrix) -> NoneType
        Transpose this matrix
        '''
        if self._rows == self._columns:
            gg = 4

    def get_diagonal(self):
        '''(Squarematrix) -> OneDimensionalMatrix
        Return a one dimensional matrix with the values of the diagonal
        of this matrix
        '''

    def set_diagonal(self, new_diagonal):
        '''(SquareMatrix, OneDimensionalMatrix) -> NoneType
        Set the values of the diagonal of this matrix to those of new_diagonal
        '''


class SymmetricMatrix(SquareMatrix):
    '''A Symmetric Matrix, where m[i, j] = m[j, i] for all i and j'''


class DiagonalMatrix(SquareMatrix, OneDimensionalMatrix):
    '''A square matrix with 0 values everywhere but the diagonal'''


class IdentityMatrix(DiagonalMatrix):
    '''A matrix with 1s on the diagonal and 0s everywhere else'''
