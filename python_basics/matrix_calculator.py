import numpy as np

def create_matrix():
    """CREATE A MATRIX FROM USER INPUT"""
    rows=int(input("ENTER THE NUMBER OF ROWS"))
    cols=int(input("ENTER THE NUMBER OF COLUMNS"))

    print(f"ENTER {rows*cols} elements (space-separated):")
    elements=list(map(float,input().split()))
    
    if len(elements)!=rows*cols:
        print("WRONG NUMBER OF ELEMENTS")
        return None
    matrix=np.array(elements).reshape(rows,cols)
    return matrix

def display_matrix(matrix,name="MATRIX"):
    """DISPLAY A MATRIC NICELY"""
    print(f"\n{name}:")
    print(matrix)
    print(f"shape: {matrix.shape}")
    print()

def add_matrices(A,B):
    """ADD TWO MATRICES"""
    if A.shape!=B.shape:
        print("ERROR:MATRICES MUST HAVE SAME SHAPE")
        return None
    return A+B

def multiply_matrices(A,B):
    """MULTIPLY TWO MATRICES"""
    if A.shape[1] !=B.shape[0]:
        print("ERROR CAN'T MULTIPLY {A.shape}x{B.shape}")
        print("COLUMNS OF A ({A.shape[1]}) must be equal rows of B({B.shape[0]})")
        return None
    return np.dot(A,B)

def transpose_matrix(A):
    """TRANSPOSE OF A MATRIX"""
    return A.T

def determinant(A,):
    """SQAUARE MATRICES ONLY"""
    if A.shape[1]!=A.shape[0]:
        print("DETERMINANT CANNOT BE CALCULATED")
        return None
    return np.linalg.det(A)

def inverse_matrix(A):
    """SQAUARE MATRICES ONLY"""
    if A.shape[1]!=A.shape[0]:
        print("DETERMINANT CANNOT BE CALCULATED")
        return None
    det=np.linalg.det(A)
    if abs(det)<1e-10:
        print("ERROR:MATRIX IS SINGULAR NOT INVERTIBLE")
        return None
    return np.linalg.inv(A)

def matrix_stats(A):
    """CALCULATE STATISTICS OF A MATRIX"""
    print(f"\n-----STATISTICS OF MATRIX-----")
    print(f"SUM={np.sum(A)}")
    print(f"MEAN={np.mean(A)}")
    print(f"Min:{np.min(A)}")
    print(f"MAX={np.max(A)}")
    print(f"STD DEVIATION={np.std(A)}")
    print()

def main():
    """MAIN CALCULATOR LOOP"""
    print("="*50)
    print("MATRIX CALCULATOR")
    print("="*50)

    while True:
        print("\n--- MENU ---")
        print("1. Add two matrices")
        print("2. Multiply two matrices")
        print("3. Transpose matrix")
        print("4. Determinant")
        print("5. Inverse matrix")
        print("6. Matrix statistics")
        print("7. Exit")

        choice=input("\nCHOOSE ONE OPTIOPN(1-7): ")

        if choice=="1":
            print("\n-----MATRIX ADDICTION-----")
            print("ENTER MATRIX A:")
            A=create_matrix()
            if A is None:
                continue
            print("\nEnter Matrix B:")
            B = create_matrix()
            if B is None:
                continue

            display_matrix(A,"MATRIX A")
            display_matrix(B,"MATRIX B")

            result=add_matrices(A,B)
            if result is not None:
                display_matrix(result,"A+B")
                input("\nPress Enter to continue...")

        elif choice=="2":
           print("\n--- Matrix Multiplication ---")
           print("Enter Matrix A:")
           A = create_matrix()
           if A is None:
                continue
            
           print("\nEnter Matrix B:")
           B = create_matrix()
           if B is None:
                continue
            
           display_matrix(A, "Matrix A")
           display_matrix(B, "Matrix B")
            
           result = multiply_matrices(A, B)
           if result is not None:
                display_matrix(result, "A x B")
                input("\nPress Enter to continue...")

        elif choice=="3":
            print("\n--- Transpose ---")
            A = create_matrix()
            if A is None:
                continue
            
            display_matrix(A, "Original Matrix")
            result = transpose_matrix(A)
            display_matrix(result, "Transposed Matrix")
            input("\nPress Enter to continue...")

            
        elif choice == "4":
            print("\n--- Determinant ---")
            A = create_matrix()
            if A is None:
                continue
            
            display_matrix(A, "Matrix")
            det = determinant(A)
            if det is not None:
                print(f"Determinant: {det:.2f}")
                input("\nPress Enter to continue...")

        elif choice == "5":
            print("\n--- Inverse ---")
            A = create_matrix()
            if A is None:
                continue
            
            display_matrix(A, "Original Matrix")
            inv = inverse_matrix(A)
            if inv is not None:
                display_matrix(inv, "Inverse Matrix")

                idenity=np.dot(A,inv)
                print("VERIFICATION (A x A inverse):")
                print(np.round(idenity,decimals=10))
                input("\nPress Enter to continue...")

        elif choice == "6":
            print("\n--- Statistics ---")
            A = create_matrix()
            if A is None:
                continue
            
            display_matrix(A, "Matrix")
            matrix_stats(A)
            input("\nPress Enter to continue...")
        
        elif choice == "7":
            print("\nGoodbye!")
            input("\nPress Enter to continue...")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()


        
        
        

                 
    
