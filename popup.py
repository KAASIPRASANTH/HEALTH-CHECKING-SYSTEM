from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
def details():
    window = Tk()
    window.title('Health Checker')
    window.geometry('400x150')

    l1 = Label(window,text='Name : ',font=(14))
    l2 = Label(window,text='Mail Id : ',font=(14))

    l1.grid(row=0,column=0,padx=5,pady=5)
    l2.grid(row=1,column=0,padx=5,pady=5)

    Name = StringVar()
    Email = StringVar()

    t1 = Entry(window,textvariable=Name,font=(14))
    t2 = Entry(window,textvariable=Email,font=(14))

    t1.grid(row=0,column=1)
    t2.grid(row=1,column=1)
    def submit():
        messagebox.showwarning(title='Information',message='Thank You!')
        window.destroy()
    def cancel():
        status = messagebox.askyesno(title='Question',message='Do you want to close the window')
        if status == True:
            window.destroy()
        else:
            messagebox.showwarning(title='Warning Message',message='Please submit!')


    b1 = Button(window,command=submit,text='Submit',font=(14))
    b2 = Button(window,command=cancel,text='Cancel',font=(14))

    b1.grid(row=2,column=1,sticky=W)
    b2.grid(row=2,column=1,sticky=E)

    window.mainloop()
    print(Name.get())

details()
