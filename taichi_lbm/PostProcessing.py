import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

@ti.data_oriented
class PostProcessingEngine:
    def __init__(self,num_curves):
        self.num_curves = num_curves
        if num_curves!=0:
            self.fig, self.axes = plt.subplots(num_curves, 1, figsize=(8, 6 * num_curves))  # 创建多个子图
            if num_curves == 1:
                self.axes = [self.axes]  # 如果只有一个子图，将其转换为列表
            plt.ion()  # 开启交互模式
    
    def update_plot(self, data_list):
        """
        更新每个子图的曲线。

        参数:
        - data_list: 一个列表，包含每个曲线的 (x, y) 数据对。
        """
        for i, (x, y) in enumerate(data_list):
            ax = self.axes[i]
            ax.clear()  # 清除当前子图的内容
            ax.plot(x, y, lw=2)  # 绘制新的曲线
            ax.set_xlabel('X')
            ax.set_ylabel('Pressure')
            ax.set_title(f'Curve {i + 1}')
            ax.grid(True)  # 添加网格

        self.fig.canvas.draw_idle()  # 强制更新图形


        
    def show(self):
        plt.show(block=False)  # 显示图形，但不阻塞程序

    def close(self):
        plt.close(self.fig)  # 关闭图形


    def nomalized_field(self,data):
        min_val=np.min(data)
        max_val=np.max(data)
        normalized_data=(data-min_val)/(max_val-min_val)
        return normalized_data
    

    def post_pressure(self,lb_field:ti.template()):
        return self.nomalized_field(lb_field.total_pressure.to_numpy())


    def post_MC_pressure(self,lb_field:ti.template()):
        images = []
        for component in range(lb_field.num_components[None]):
            data = self.nomalized_field(lb_field.pressure.to_numpy()[:,:,component])
            images.append(data)
        return np.concatenate(images, axis=1)

    def post_denstiy(self,lb_field:ti.template()):
        data=self.nomalized_field(lb_field.rho.to_numpy()[:,:,0])
        return data

    def post_vel(self,lb_field:ti.template()):
        vel = lb_field.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        return self.nomalized_field(vel_mag)
    

    def writeVTK(self,fname,lb_field:ti.template(),Tflag=None):
        rho=lb_field.total_rho.to_numpy().T.flatten()  
        vel=lb_field.vel.to_numpy()
        velx=vel[:,:,0].T.flatten()  
        vely=vel[:,:,1].T.flatten()  

        # pressure=lb_field.total_pressure.to_numpy().T.flatten()  

        T=lb_field.T.to_numpy().T.flatten()
        x_coords = np.arange(lb_field.NX)  
        y_coords = np.arange(lb_field.NY)  
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  

        x_flat = x_mesh.flatten()  
        y_flat = y_mesh.flatten()  


        filename = fname + ".vtk"  
        with open(filename, 'w') as fout:  
            fout.write("# vtk DataFile Version 3.0\n")  
            fout.write("Hydrodynamics representation\n")  
            fout.write("ASCII\n\n")  
            fout.write("DATASET STRUCTURED_GRID\n")  
            fout.write(f"DIMENSIONS {lb_field.NX} {lb_field.NY} 1\n")  
            fout.write(f"POINTS {lb_field.NX*lb_field.NY} double\n")  
          
            np.savetxt(fout, np.column_stack((x_flat, y_flat, np.zeros_like(x_flat))), fmt='%.0f')  
          
            fout.write("\n")  
            fout.write(f"POINT_DATA {lb_field.NX*lb_field.NY}\n")  


            fout.write("SCALARS density double\n")  
            fout.write("LOOKUP_TABLE density_table\n")  
            np.savetxt(fout, rho * lb_field.C_rho, fmt='%.8f') 
        
            # fout.write("SCALARS Pressure double\n")  
            # fout.write("LOOKUP_TABLE Pressure_table\n")  
            # np.savetxt(fout, pressure * lb_field.C_pressure, fmt='%.8f') 

            if Tflag is not None:
                fout.write("SCALARS T double\n")  
                fout.write("LOOKUP_TABLE T_table\n")  
                np.savetxt(fout, T, fmt='%.8f') 


            fout.write("VECTORS velocity double\n")  
            velocity_data = np.column_stack((velx * lb_field.Cu, vely * lb_field.Cu, np.zeros_like(velx)))  
            np.savetxt(fout, velocity_data, fmt='%.8f') 

        print(filename)


    def writeVTKpsi(self,fname,lb_field:ti.template()):
        PSI=lb_field.sc_field.psi_field.to_numpy()[:,:,0].T.flatten()

        x_coords = np.arange(lb_field.NX+2)  
        y_coords = np.arange(lb_field.NY+2)  
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  

        x_flat = x_mesh.flatten()  
        y_flat = y_mesh.flatten()  


        filename = fname + ".vtk"  
        with open(filename, 'w') as fout:  
            fout.write("# vtk DataFile Version 3.0\n")  
            fout.write("Hydrodynamics representation\n")  
            fout.write("ASCII\n\n")  
            fout.write("DATASET STRUCTURED_GRID\n")  
            fout.write(f"DIMENSIONS {lb_field.NX+2} {lb_field.NY+2} 1\n")  
            fout.write(f"POINTS {(lb_field.NX+2)*(lb_field.NY+2)} double\n")  
          
            np.savetxt(fout, np.column_stack((x_flat, y_flat, np.zeros_like(x_flat))), fmt='%.0f')  
          
            fout.write("\n")  
            fout.write(f"POINT_DATA {(lb_field.NX+2)*(lb_field.NY+2)}\n")  



            fout.write("SCALARS PSI double\n")  
            fout.write("LOOKUP_TABLE T_table\n")  
            np.savetxt(fout, PSI, fmt='%.8f') 



        print(filename)
