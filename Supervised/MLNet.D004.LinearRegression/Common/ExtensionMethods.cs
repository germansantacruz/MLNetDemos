﻿using System;
using System.Linq;

namespace MLNet.D004.LinearRegression.Common
{
    public static class ExtensionMethods
    {
        public static string[] ToPropertyList<T>(this Type objType, string labelName)
        {
            return objType.GetProperties()
                .Where(a => a.Name != labelName)
                .Select(a => a.Name)
                .ToArray();
        }            
    }
}
