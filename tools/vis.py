import numpy as np
from sympy.physics.mechanics import ReferenceFrame,Point,Vector
from sympy import symbols
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Text
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Line3D,Line3DCollection
from stl import mesh
import copy

def transformationMatrix(rotation,translation):
    '''Builds a 4x4 transformation matrix form rotation and translation'''
    translation=np.reshape(translation,(3,1))
    upper=np.concatenate((rotation,translation),axis=1)
    lower=np.array([[0,0,0,1]])
    return np.concatenate((upper,lower))
    
class Visualizer:
    textoffset=np.transpose(np.array([[0.1,0.1,0.1]]))
    def __init__(self,baseFrame,origin):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.baseFrame=baseFrame
        self.origin=origin
        self.objs=[] #Objects hold all the drawables as (matrixeqs,obj,params)
        self.xrange=[-1,1]
        self.yrange=[-1,1]
        self.zrange=[-1,1]
        plt.show()
        
    def add(self,frame,point,shape=None,frame_scale=1,mesh_scale=1):
        '''Add an actor consisting of a frame a point and
        optionally a shape given by an stl filepath'''
        
        p=point.pos_from(self.origin).to_matrix(self.baseFrame)
        f=frame.dcm(self.baseFrame)
        
        # Will append the dict with matrixeqs,obj,params                
        # This means: create the matplotlib obj, compute the eqs, and fill params
        if shape==None:
            # Point
            actor,=self.ax.plot3D(0,0,0,'b.')
            obj=dict()
            obj['actor']=actor
            obj['eq']=p                
            self.objs.append(obj)
            
            # Text for the point
            actor=self.ax.text2D(0,0,point.name)        
            obj=dict()
            obj['actor']=actor
            obj['eq']=p                
            self.objs.append(obj)
                    
            # Quiver for the frame             
            actor=list()
            for i in range(0,3):
                actor.append(self.ax.quiver3D(0,0,0,0,0,0))
            obj=dict()
            obj['actor']=actor
            obj['eq']=(p,f)
            obj['scale']=frame_scale        
            self.objs.append(obj)
        else:
            # Stl mesh
            obj=dict()
            obj['mesh']=None
            if shape!=None:
                shape_mesh = mesh.Mesh.from_file(shape)
                shape_mesh.vectors=shape_mesh.vectors*mesh_scale
                obj['actor']=shape_mesh
                obj['eq']=(p,f)
                obj['scale']=mesh_scale
                self.objs.append(obj)
                        
    def plot(self,replacements=dict()):
        '''Collect all the objects and redraw'''            
        for obj in self.objs:          
            # Do a first pass skipping text since it uses projection
            # and we don't have the autoscaling baked yet
            if (isinstance(obj['actor'],Text)):                
                continue                                
            elif (isinstance(obj['actor'],Line3D)):
                p=obj['eq'].subs(replacements)
                p=np.array(p,dtype=np.float64) 
                obj['actor'].set_data_3d(p[0],p[1],p[2])
                self.autoscale(p,boundary=0)
            elif (isinstance(obj['actor'],list)): #Frame
                colors=['r','g','b']
                p=obj['eq'][0].subs(replacements)
                p=np.array(p,dtype=np.float64) 
                f=obj['eq'][1].subs(replacements)
                f=np.array(f,dtype=np.float64) 
                for i in range(0,3):                
                    obj['actor'][i].remove()                    
                    obj['actor'][i]=self.ax.quiver(p[0],p[1],p[2],f[i,0],f[i,1],f[i,2],length=obj['scale'],normalize=False,color=colors[i])                                
                self.autoscale(np.squeeze(p),boundary=obj['scale'])
            elif (isinstance(obj['actor'],mesh.Mesh)): #Mesh
                if 'surf' in obj:
                    obj['surf'].remove()
                p=obj['eq'][0].subs(replacements)
                p=np.array(p,dtype=np.float64) 
                f=obj['eq'][1].subs(replacements)
                f=np.array(f,dtype=np.float64)
                H=transformationMatrix(np.transpose(f),p)
                transformedMesh=copy.deepcopy(obj['actor'])
                transformedMesh.transform(H)
                obj['surf']=self.ax.add_collection3d(mplot3d.art3d.Poly3DCollection(transformedMesh.vectors))
                obj['surf'].set_edgecolor(np.array([0.2,0.2,0.2,0.05],dtype=np.float64))
                obj['surf'].set_facecolor(np.array([0.2,0.2,0.2,0.2],dtype=np.float64))
                self.autoscale_mesh(transformedMesh)         
        
        #Autoscale ax box
        self.ax.set_xlim(xmin=self.xrange[0],xmax=self.xrange[1])      
        self.ax.set_ylim(ymin=self.yrange[0],ymax=self.yrange[1])      
        self.ax.set_zlim(zmin=self.zrange[0],zmax=self.zrange[1])      
        self.ax.set_box_aspect([self.xrange[1]-self.xrange[0],self.yrange[1]-self.yrange[0],self.zrange[1]-self.zrange[0]])
        
        for obj in self.objs:          
            if (isinstance(obj['actor'],Text)):                                
                p=obj['eq'].subs(replacements)
                p=np.array(p,dtype=np.float64)+Visualizer.textoffset
                x, y, _ = proj3d.proj_transform(p[0],p[1],p[2],self.ax.get_proj())                
                obj['actor'].set_position((x,y)) 
            else:
                continue
                
    def autoscale_mesh(self,mesh):
        '''Computes the axis range based on mesh points data'''
        pmin = np.min(mesh.vectors,axis=(0,1))
        pmax = np.max(mesh.vectors,axis=(0,1))        
        self.autoscale(pmin)
        self.autoscale(pmax)
        
    def autoscale(self,p,boundary=0):
        if p[0]+boundary>self.xrange[1]:
            self.xrange[1]=p[0]+boundary
        elif p[0]-boundary<self.xrange[0]:
            self.xrange[0]=p[0]-boundary
        if p[1]+boundary>self.yrange[1]:
            self.yrange[1]=p[1]+boundary
        elif p[1]-boundary<self.zrange[0]:
            self.yrange[0]=p[1]-boundary
        if p[2]+boundary>self.zrange[1]:
            self.zrange[1]=p[2]+boundary
        elif p[2]-boundary<self.zrange[0]:
            self.zrange[0]=p[2]-boundary 
        #Make scaling equal on xyz
        #self.xrange[0]=np.min([self.xrange[0],self.yrange[0],self.xrange[0]])
        #self.xrange[1]=np.max([self.xrange[1],self.yrange[1],self.xrange[1]])
        #self.yrange=self.xrange
        #self.zrange=self.xrange
        