
#from Tkinter import *
import tkinter as tk
import tkinter.ttk
import threading
import signal
import time
import random

PLAY= False

class GUI(threading.Thread):
    def __init__(self, instr_ls_obj):

        self.instr_ls_obj= instr_ls_obj
        threading.Thread.__init__(self)

        self.start()

    def callback(self):
        self.app.destroy()
        self.app.quit()
    
    def animate(self):
        DONE= self.app.cluster.animate()
#        DONE= True
        if not DONE:
          self.app.after(10, self.animate)

    def run(self):
        self.app = animation(self, self.instr_ls_obj)
        self.app.after(1000, self.animate)
        self.app.mainloop()

class animation(tk.Tk):
  
    def __init__(self, master, instr_ls_obj, *args, **kwargs):
        self.master= master

        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_geometry("%dx%d+10+10" % (1000,700))
        self.config(bg="white")
        

        if 'rate' in kwargs:
          self.update_rate= kwargs['rate']

        #self.attributes("-fullscreen", True) 
        
        self.menu= menu_frame(self)
        self.menu.pack()

        self.cluster= cluster_frame(self, instr_ls_obj, delay= 0.1)
        self.cluster.pack()
        

class menu_frame(tk.Frame):
  def __init__(self, master):
    self.master= master
    tk.Frame.__init__(self,master, cursor="arrow", height= 10, width= 10)
    
    
#    im= PIL.Image.open("./src/gui_images/play_button.png")
    #play_image= PIL.ImageTk.PhotoImage(im)
    play_image= tk.PhotoImage(file="./src/gui_images/pause.gif")
    
    self.play_pause_button= tk.Button(self, text= "play/pause", command= self.play_pause_toggle, height= 10, width=10)
#    self.play_pause_button= tk.Button(self, image= play_image, command= self.play_pause_toggle, height= 100, width=100)
    self.play_pause_button.grid(row=0, column=0) 
    
    self.play_pause_button= tk.Button(self, text= "speed++", command= self.speed_incr, height= 10, width=10)
    self.play_pause_button.grid(row=0, column=1) 

    self.play_pause_button= tk.Button(self, text= "speed--", command= self.speed_decr, height= 10, width=10)
    self.play_pause_button.grid(row=0, column=2) 
  
  def speed_incr(self):
    self.master.cluster.delay /= 1.5

  def speed_decr(self):
    self.master.cluster.delay *= 1.5

  def play_pause_toggle(self):
    global PLAY
    if PLAY:
      PLAY= False
    else:
      PLAY= True
    

class cluster_frame(tk.Frame):
    def __init__(self, master, instr_ls_obj, delay=1, height=500,width=200, D_TREE= 3, N_TREE=4, BANK_DEPTH= 18):
        self.master= master
        tk.Frame.__init__(self,master, cursor="arrow",relief=tk.SUNKEN,bd=2)
        
        self.delay=delay
        
        self.BANK_DEPTH= BANK_DEPTH

        PE_SIZE= 25
        self.N_BANK= N_TREE * (2**(D_TREE))
        
        self.tree= {}
        # Create trees
        for i in range(N_TREE):
          self.tree[i]= tree_canvas(self, D=D_TREE, PE_SIZE= PE_SIZE);
          self.tree[i].grid(row=0, column=i+1)
        


        # reg banks
        self.reg_banks= reg_banks(self, REG_W= PE_SIZE, BANK_DEPTH= BANK_DEPTH)
        self.reg_banks.grid(row=2, column=1, columnspan= N_TREE)
        
        # Arrows
        self.arrows_cluster= arrow_canvas(self)
        self.arrows_cluster.grid(row=1, column=2, rowspan= 1, columnspan=2)


        self.arrows_ld= arrow_canvas(self, arrow_col= 'blue')
        self.arrows_ld.grid(row=3, column=2)

        self.arrows_st= arrow_canvas(self, arrow_head= tk.LAST, arrow_col= 'green')
        self.arrows_st.grid(row=3, column=3)
        
        # Memory
        self.mem= rect_canvas(self)
        self.mem.grid(row=4, column=1,columnspan=N_TREE)

        self.instr_ls_obj= instr_ls_obj
    
    def reset_frame(self):
      self.arrows_cluster.disable_arrow()
      self.arrows_ld.disable_arrow()
      self.arrows_st.disable_arrow()
      self.mem.rect_none()
      
      for tree in list(self.tree.values()):
        tree.disable_tree()

      self.master.update()
    
    def init_reg_bank(self):
      for bank in range(self.N_BANK):
        for reg in range(self.BANK_DEPTH):
          self.reg_banks.reg_occupied(bank, reg)
          if random.randint(1, self.BANK_DEPTH) < 3:
            break
    def reset_reg_bank(self):
      for bank in range(self.N_BANK):
        for reg in range(self.BANK_DEPTH):
          self.reg_banks.reg_empty(bank, reg)

    def bb_instr(self):
      self.reset_reg_bank()
      self.init_reg_bank()
      self.reset_frame()
      for tree in list(self.tree.values()):
        tree.enable_tree()

      for bank in range(self.N_BANK):
        self.reg_banks.reg_read(bank, random.randint(1,self.BANK_DEPTH/3))
      
      self.arrows_cluster.enable_arrow()
      self.master.update()
    
    def ld_instr(self):
      self.reset_reg_bank()
      self.reset_frame()
      self.init_reg_bank()
      self.arrows_ld.enable_arrow()
      self.mem.rect_ld()
      for bank in range(self.N_BANK):
        self.reg_banks.reg_ld(bank, random.randint(1,self.BANK_DEPTH/3))
      self.master.update()

    def st_instr(self):
      self.reset_reg_bank()
      self.reset_frame()
      self.init_reg_bank()
      self.arrows_st.enable_arrow()
      self.mem.rect_st()
      for bank in range(self.N_BANK):
        self.reg_banks.reg_st(bank, random.randint(1,self.BANK_DEPTH/3))
      self.master.update()

    def wait(self, state):
      global PLAY
      while PLAY== state:
        time.sleep(0.5)
        self.master.update()

    def animate(self):
      instr_ls_obj= self.instr_ls_obj
      
      self.wait(False)
      self.bb_instr()

      self.wait(True)
      self.ld_instr()

      self.wait(False)
      self.st_instr()
      self.wait(True)
      
      self.reset_reg_bank()

      global PLAY
      for curr_instr_idx, curr_instr in enumerate(instr_ls_obj.instr_ls):
        self.reset_frame()
        self.wait(False)
        print(curr_instr.name, end=' ') 
        if curr_instr.is_type('bb'):
          self.arrows_cluster.enable_arrow()
          for tree in list(self.tree.values()):
            tree.enable_tree()

          for node, node_details in list(curr_instr.in_node_details_dict.items()):
            self.reg_banks.reg_read(node_details.bank, node_details.pos)
          self.master.update()
          time.sleep(self.delay)

          for node, node_details in list(curr_instr.out_node_details_dict.items()):
            self.reg_banks.reg_write(node_details.bank, node_details.pos)
          self.master.update()
          time.sleep(self.delay)
          
          # Restore the color
          for node, node_details in list(curr_instr.in_node_details_dict.items()):
            self.reg_banks.reg_occupied(node_details.bank, node_details.pos)
          for node, node_details in list(curr_instr.fully_consumed_node_details_dict.items()):
            self.reg_banks.reg_empty(node_details.bank, node_details.pos)
          for node, node_details in list(curr_instr.out_node_details_dict.items()):
            self.reg_banks.reg_occupied(node_details.bank, node_details.pos)
          self.master.update()
          time.sleep(self.delay)

        if curr_instr.is_type('ld'):
          self.arrows_ld.enable_arrow()
          self.mem.rect_ld()
          for node, node_details in list(curr_instr.node_details_dict.items()):
            self.reg_banks.reg_ld(node_details.bank, node_details.pos)
          self.master.update()
          time.sleep(self.delay)

          for node, node_details in list(curr_instr.node_details_dict.items()):
            self.reg_banks.reg_occupied(node_details.bank, node_details.pos)

        if curr_instr.is_type('st'):
          self.arrows_st.enable_arrow()
          self.mem.rect_st()
          for node, node_details in list(curr_instr.node_details_dict.items()):
            self.reg_banks.reg_st(node_details.bank, node_details.pos)
          self.master.update()
          time.sleep(self.delay)

          for node, node_details in list(curr_instr.node_details_dict.items()):
            self.reg_banks.reg_empty(node_details.bank, node_details.pos)

          

        self.master.update()
      
      time.sleep(1)
      
      return True

      if len(instr_ls_obj.instr_ls) == self.curr_instr_idx:
        return True
      else:
        return False

class rect_canvas(tk.Canvas):
    def __init__(self, master=None):
      self.bg_color= 'white'
      self.height= 100
      self.width= 500

      tk.Canvas.__init__(self, master, height= self.height, width= self.width, bg=self.bg_color)
      self.rect= self.create_rectangle(0, 0, self.width, self.height)
      self.rect_ld()

    def rect_ld(self):
      self.itemconfig(self.rect, fill='blue')
    def rect_st(self):
      self.itemconfig(self.rect, fill='green')
    def rect_none(self):
      self.itemconfig(self.rect, fill='white')

class arrow_canvas(tk.Canvas):
    def __init__(self, master=None, arrow_head=tk.FIRST, arrow_col= 'red'):
      self.bg_color= 'white'
      self.arrow_col= arrow_col

      tk.Canvas.__init__(self, master, height= 60, width= 100, bg=self.bg_color)

      self.up_arrow= self.create_line(50,0,50,50, arrow=arrow_head, width= 20, arrowshape= (10,14,10), disabledfill= self.bg_color, fill=self.arrow_col)
      self.disable_arrow()
#      self.enable_arrow()
    
    def disable_arrow(self):
      self.itemconfig(self.up_arrow, state= tk.HIDDEN)

    def enable_arrow(self):
      self.itemconfig(self.up_arrow, state= tk.NORMAL)

class reg_banks(tk.Canvas):
    def __init__(self, master=None, REG_H= 10, REG_W=20, N_BANK= 32, BANK_DEPTH= 16 ):
        self.color="White"
        tk.Canvas.__init__(self,master,bg=self.color, cursor="arrow", height= REG_H*BANK_DEPTH, width= REG_W*N_BANK)
        
        self.reg={}

        for i in range(BANK_DEPTH):
          self.reg[i]= {}
          for j in range(N_BANK):
            self.reg[i][j]= self.create_rectangle(j*REG_W, i*REG_H, (j+1)* REG_W, (i+1)*REG_H);
            self.reg_empty(i,j)
        
        # Just invert indices
        reg_copy= {}
        for i in range(N_BANK):
          reg_copy[i]= {}
          for j in range(BANK_DEPTH):
            reg_copy[i][j]= self.reg[j][i]

        self.reg= reg_copy

    def reg_occupied(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'yellow')
    
    def reg_read(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'red')

    def reg_write(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'pink')
    
    def reg_ld(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'blue')

    def reg_st(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'green')

    def reg_empty(self, bank, pos):
      self.itemconfig(self.reg[bank][pos], fill= 'white')

class tree_canvas(tk.Canvas):     
    def __init__(self, master=None, PE_SIZE= 25, D=3):
        self.color="White"
        height= (D+1)* 1.5*PE_SIZE 
        width= PE_SIZE * (2**D) 
        tk.Canvas.__init__(self,master,bg=self.color, cursor="arrow", height= height, width= width)

        # dict of dict to store PE circle object
        # key: depth, pos. Val= oval obj
        self.PE= {}
        
        size= PE_SIZE
#        assert width > (2**(D-1))*(size+10), "Assert width too small"
#        assert height > D*1.25*size

        # Draw circles
        self.center={}
        y_off= 0
        for i in range(D):
          self.PE[i]= {}
          self.center[i]= {}
          y_off += 1.5*size
          for j in range(2**i):
            x_off= (float((j)) * width/(2**(i))) + width/(2**(i+1))

            self.PE[i][j]= self.create_oval(x_off - (0.5*size),y_off - size*0.5, x_off + (size*0.5), y_off + size*0.5 )
            
            self.center[i][j]= [x_off, y_off]

        # draw arrows
        for i in range(1,D):
          for j in range(2**i):
            x0= self.center[i][j][0] #+ ((-1)**(j)) * 0.5*size
            y0= self.center[i][j][1] - 0.5*size

            x1= float(self.center[i-1][j/2][0] + ((-1)**(j+1)) * 0.5*size)
            y1= float(self.center[i-1][j/2][1] + 0.5*size)
            
            self.create_line(x0,y0,x1,y1, arrow= tk.LAST )
    
    def enable_tree(self):
      for row in list(self.PE.values()):
        for PE in list(row.values()):
          self.itemconfig(PE, fill= 'red')
    
    def disable_tree(self):
      for row in list(self.PE.values()):
        for PE in list(row.values()):
          self.itemconfig(PE, fill= 'white')

