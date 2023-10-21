import cv2
import numpy as np

#copyright intellar@intellar.ca

def disk(h, w):
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask_ = dist_from_center <= radius
    mask = np.zeros((h,w))
    mask[mask_]=1

    return mask

def sanitize_det(DetA, Dx2,Dy2,DxDy, threshold_sol_determinant,threshold_ratio_eigenvalues):

    DetA[np.abs(DetA)<threshold_sol_determinant] = np.inf
    #check eigenvalue ratio,this remove response on single straight edges    
    S1 = 1/2*Dy2+1/2*Dx2+1/2*(Dy2**2-2*Dx2*Dy2+Dx2**2+4*DxDy**2)**(1/2)
    S2 = 1/2*Dy2+1/2*Dx2-1/2*(Dy2**2-2*Dx2*Dy2+Dx2**2+4*DxDy**2)**(1/2)
    #avoid divide by zero
    S2[np.abs(S2)<1e-10] = np.inf  
    S1S2 = S1/S2
    DetA[ S1S2<threshold_ratio_eigenvalues ] = np.inf
    DetA[ S1S2>1/threshold_ratio_eigenvalues ] = np.inf
    return DetA
    
def alloc_matrix_list(width, height):    
    return [ [ [] for i in range(width) ] for j in range(height) ]

def ASN_detector(img,threshold_min_nb_solutions = 20, threshold_sol_determinant = 1e-6, threshold_ratio_eigenvalues = 0.1, config_sobel_kernel_size=7, config_integration_for_solution_size=25):
        
    use_bilinear_fit = 0

    integration_windows = disk(config_integration_for_solution_size,config_integration_for_solution_size)
    integration_windows /= np.sum(integration_windows.flatten())

    ddepth = cv2.CV_32F
    scale = 1
    delta = 0
    dx = cv2.Sobel(img, ddepth, 1, 0, ksize=config_sobel_kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    dy = cv2.Sobel(img, ddepth, 0, 1, ksize=config_sobel_kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    dxdy = dx*dy
    dx2 = dx*dx 
    dy2 = dy*dy
    ddepth = -1

    Dx2 = cv2.filter2D(dx*dx,ddepth,integration_windows)
    Dy2 = cv2.filter2D(dy*dy,ddepth,integration_windows)
    DxDy = cv2.filter2D(dxdy,ddepth,integration_windows)
    DxDyX = cv2.filter2D(dxdy*x,ddepth,integration_windows)
    DxDyY = cv2.filter2D(dxdy*y,ddepth,integration_windows)
    Dx2X = cv2.filter2D(dx2*x,ddepth,integration_windows)
    Dy2Y = cv2.filter2D(dy2*y,ddepth,integration_windows)

    DetA = Dx2*Dy2-DxDy**2
    DetA = sanitize_det(DetA,Dx2,Dy2,DxDy, threshold_sol_determinant,threshold_ratio_eigenvalues)
    
    localSolutionX = (Dy2*(Dx2X+DxDyY)-DxDy*(Dy2Y+DxDyX))/(DetA)
    localSolutionY = (Dx2*(Dy2Y+DxDyX)-DxDy*(Dx2X+DxDyY))/(DetA)

    accumulateur = np.zeros(img.shape)
    accumulateurXY = alloc_matrix_list(img.shape[1],img.shape[0])
    convergence_regions = np.zeros(img.shape)

    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sol_x = localSolutionX[i,j]
            sol_y = localSolutionY[i,j]
            if np.isfinite(sol_x) and np.isfinite(sol_y):
                if sol_x != 1.0 and sol_y != 1.0 and np.abs(sol_x)>0.1  and np.abs(sol_y)>0.1:
                    
                    sol_x_i = int(sol_x)
                    sol_y_i = int(sol_y)
                    sol_x_f = sol_x-sol_x_i
                    sol_y_f = sol_y-sol_y_i

                    

                    if sol_x>1 and sol_x<img.shape[1]-1 and sol_y>1 and sol_y<img.shape[0]-1:
                        #accumulateur[sol_y,sol_x] += 1
                        # p1--------p2
                        #  |----X----|
                        #  |    |    |
                        # p3--------p4
                        accumulateur[sol_y_i,sol_x_i]   += (1-sol_x_f)*(1-sol_y_f)
                        accumulateur[sol_y_i+1,sol_x_i] += (1-sol_x_f)*(sol_y_f)
                        accumulateur[sol_y_i,sol_x_i+1] += (sol_x_f)*(1-sol_y_f)
                        accumulateur[sol_y_i+1,sol_x_i+1] += (sol_x_f)*(sol_y_f)

                        accumulateurXY[sol_y_i][sol_x_i].append((j,i))
                        


    #find local max, find subpixel by fitting quadric on accumulator values. max is where derivatives are 0
    #build quadric  ax2+bxy+cy2+dx+ey+c=vals
    A = []
    positions = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]    
    for p in positions:
        #ax2+bxy+cy2+dx+ey+c
        A.append([ p[0]**2, p[0]*p[1], p[1]**2, p[0], p[1], 1  ])
    A = np.array(A)
    iAtA_At = np.linalg.inv(A.T@A)@A.T

    
    pts = []
    for i in range(1,accumulateur.shape[0]-1):
        for j in range(1,accumulateur.shape[1]-1):            
            sub_img = accumulateur[i-1:i+2,j-1:j+2]
            #enough solution for this position?
            if sub_img[1,1]>threshold_min_nb_solutions:            
                mmax = sub_img[1,1]>sub_img            
                mmax[1,1]=True
                #is a local maximum?
                if mmax.all():
                    
                    
                    
                    if use_bilinear_fit==0:
                        #create a new solution with all contributors
                        dx2_ = 0
                        dx2x_ = 0
                        dy2_ = 0
                        dy2y_ = 0                  
                        dxdy_ = 0
                        dxdyx_ = 0
                        dxdyy_ = 0


                        for ii in range(i-1,i+2):
                            for jj in range(j-1,j+2):
                                dd = accumulateurXY[ii][jj]
                                for cc_to_add in accumulateurXY[ii][jj]:
                                    #use integrated matrix, will give more importance to redundant region of the mask
                                    dx2_ += Dx2[cc_to_add[1],cc_to_add[0]]
                                    dx2x_ += Dx2X[cc_to_add[1],cc_to_add[0]]
                                    dy2_ += Dy2[cc_to_add[1],cc_to_add[0]]
                                    dy2y_ += Dy2Y[cc_to_add[1],cc_to_add[0]]
                                    dxdy_ += DxDy[cc_to_add[1],cc_to_add[0]]
                                    dxdyx_ += DxDyX[cc_to_add[1],cc_to_add[0]]
                                    dxdyy_ += DxDyY[cc_to_add[1],cc_to_add[0]]
                                    convergence_regions[cc_to_add[1],cc_to_add[0]] = 255

                        detA_ = dx2_*dy2_-dxdy_**2                    
                        solution_X = (dy2_*(dx2x_+dxdyy_)-dxdy_*(dy2y_+dxdyx_))/(detA_)
                        solution_Y = (dx2_*(dy2y_+dxdyx_)-dxdy_*(dx2x_+dxdyy_))/(detA_)
                        pts.append([solution_X,solution_Y])
                        
                    #bilinear fit
                    if use_bilinear_fit==1:
                        b = np.array(sub_img.flatten())
                        quadric = iAtA_At@b
                        a_ = quadric[0]
                        b_ = quadric[1]
                        c_ = quadric[2]
                        d_ = quadric[3]
                        e_ = quadric[4]
                        f_ = quadric[5]
                        den = (4*a_*c_-b_**2)
                        x = [ (b_*e_-2*c_*d_) / den, (b_*d_-2*a_*e_) / den ]
                        fitted_solution = [j+x[0],i+x[1]]
                        pts.append(fitted_solution)




    pts = np.array(pts)


    
    

    return pts, accumulateur, convergence_regions

